#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup

tf.logging.set_verbosity(tf.logging.INFO)

BUCKET = None  # set from task.py
PATTERN = 'of' # gets all files

# Determine CSV, label, and key columns
#CSV_COLUMNS = 'example_id,subreddit,comment,score,comment_ints'.split(',')
CSV_COLUMNS = 'example_id,subreddit,comment,comment_ints,score'.split(',')
LABEL_COLUMN = 'score'
KEY_COLUMN = 'example_id'

# Set default values for each CSV column
#DEFAULTS = [['nokey'], ['null'], ['null'], [0.0], [[0.0],[0.0],[0.0],[0.0],[0.0]] ]
DEFAULTS = [['nokey'], ['null'], ['null'], [[0],[0]], [0.0] ]

# Define some hyperparameters
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 512
NEMBEDS = 3
NNSIZE = [64, 16, 4]

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(prefix, mode, batch_size):

    def _input_fn():

        def decode_csv(value_column):

            columns = tf.decode_csv(value_column, field_delim='|', record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
        
        # Use prefix to create file path
        file_path = '{}/{}*{}*'.format(INPUT_DIR, prefix, PATTERN)

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(file_path)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                    .map(decode_csv))  # Transform each elem by applying decode_csv fn
      
        if mode == tf.estimator.ModeKeys.TRAIN:

            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)

        else:

            num_epochs = 1 # end-of-input after this
 
        dataset = dataset.repeat(num_epochs).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


# Define feature columns
def get_wide_deep():

    MAX_DOCUMENT_LENGTH = 20  
    PADWORD = 'xyzpadxyz'
    EMBEDDING_SIZE = 10
    VOCAB_SIZE = 10002

    # Define column types
    subreddit = tf.feature_column.categorical_column_with_vocabulary_list('subreddit', ['news', 'ireland', 'pics'])

    vocab_table = lookup.index_table_from_file(
        vocabulary_file='{}/vocab.csv-00000-of-00001'.format(INPUT_DIR), 
        num_oov_buckets=1, 
        vocab_size=None, 
        default_value=-1
        )

    # i have tried all these and none work
    #comment_words = tf.string_split(tf.get_variable('comment'))
    #comment_words = tf.string_split(['comment'])
    #comment_words = tf.string_split(tf.constant(['comment']))
    #comment_words = tf.string_split([comment])
    #comment_words = tf.string_split('comment')
    #comment = tf.constant(['comment'])
    #comment = tf.constant(dataset['comment'])
    #comment = tf.constant(comment)
    #comment_words = tf.string_split(features["comment"])
    #comment_words = tf.string_split(dataset["comment"])
    #comment = tf.constant(features.get('comment'))
    #comment = tf.constant(features.get('comment'))
    
    #comment_words = tf.string_split(features.get("comment"))
    #comment_words = tf.string_split(dataset.get("comment"))

    #comment_densewords = tf.sparse_tensor_to_dense(comment_words, default_value=PADWORD)
    #comment_numbers = vocab_table.lookup(comment_densewords)
    #comment_padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    #comment_padded = tf.pad(comment_numbers, comment_padding)
    #comment_sliced = tf.slice(comment_padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])

    #print('comment_sliced={}'.format(comment_words))  # (?, 20)
    #comment_integerized = tf.contrib.layers.sparse_column_with_integerized_feature(comment_sliced, bucket_size=VOCAB_SIZE, combiner='sum')
    
    #comment_bow = tf.one_hot(comment_sliced)
    
    #comment_embeds = tf.contrib.layers.embedding_column(comment_integerized, dimension=EMBEDDING_SIZE)
    #print('comment_embeds={}'.format(comment_embeds)) # (?, 20, 10)  
    
    # Sparse columns are wide, have a linear relationship with the output
    wide = [ subreddit ]
        
    # Continuous columns are deep, have a complex relationship with the output
    deep = [ ]

    return wide, deep


# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():

    feature_placeholders = {
        'subreddit': tf.placeholder(tf.string, [None]),
        #'comment': tf.placeholder(tf.string, [None]),
        KEY_COLUMN: tf.placeholder_with_default(tf.constant(['nokey']), [None])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# create metric for hyperparameter tuning
def my_rmse(labels, predictions):

    pred_values = predictions['predictions']
    
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}


# forward to key-column to export
def forward_key_to_export(estimator):

    estimator = tf.contrib.estimator.forward_features(estimator, KEY_COLUMN)
    # return estimator

    ## This shouldn't be necessary (I've filed CL/187793590 to update extenders.py with this code)
    config = estimator.config


    def model_fn2(features, labels, mode):

        estimatorSpec = estimator._call_model_fn(features, labels, mode, config=config)

        if estimatorSpec.export_outputs:
          
            for ekey in ['predict', 'serving_default']:

                estimatorSpec.export_outputs[ekey] = tf.estimator.export.PredictOutput(estimatorSpec.predictions)

        return estimatorSpec

    return tf.estimator.Estimator(model_fn=model_fn2, config=config)


# Create estimator to train and evaluate
def train_and_evaluate(output_dir):

    wide, deep = get_wide_deep()
    
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = output_dir,
        linear_feature_columns = wide,
        dnn_feature_columns = deep,
        dnn_hidden_units = NNSIZE)
    
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    
    estimator = forward_key_to_export(estimator)

    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset('train', tf.estimator.ModeKeys.TRAIN, BATCH_SIZE),
        max_steps = TRAIN_STEPS)
    
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset('eval', tf.estimator.ModeKeys.EVAL, 2**15),  # no need to batch in eval
        steps = EVAL_STEPS,
        start_delay_secs = 60, # start evaluating after N seconds
        throttle_secs = 300,  # evaluate every N seconds
        exporters = exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
