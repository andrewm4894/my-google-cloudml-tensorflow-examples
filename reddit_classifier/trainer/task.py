"""Sample for Reddit dataset can be run as a wide or deep model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import json
import math
import os
import sys

import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.estimator import forward_features

MODEL_TYPES = ['linear', 'deep', 'deep_classifier']
LINEAR, DEEP, DEEP_CLASSIFIER = MODEL_TYPES

KEY_FEATURE_COLUMN = 'example_id'

TARGET_FEATURE_COLUMN = 'score'
TARGET_FEATURE_LABELS = [0,1]

CROSS_HASH_BUCKET_SIZE = int(1e6)

MODEL_DIR = 'model'

COLUMN_NAMES = [
    'subreddit_id', 'toplevel', 'author_bow', 'comment_body_bow',
    'comment_parent_body_bow'
]

#COLUMN_NAMES = [
#    'subreddit_id'
#]


def create_parser():
  """Initialize command line parser using arparse.
  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_type',
      help='Model type to train on',
      choices=MODEL_TYPES,
      default=LINEAR)
  parser.add_argument(
      '--train_data_paths', type=str, action='append', required=True)
  parser.add_argument(
      '--eval_data_paths', type=str, action='append', required=True)
  parser.add_argument('--output_path', type=str, required=True)
  # The following three parameters are required for tf.Transform.
  parser.add_argument('--raw_metadata_path', type=str, required=True)
  parser.add_argument('--transformed_metadata_path', type=str, required=True)
  parser.add_argument('--transform_savedmodel', type=str, required=True)
  parser.add_argument(
      '--hidden_units',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`64 32` means first layer has 64 nodes and second one has 32.',
      default=[512],
      type=int)
  parser.add_argument(
      '--batch_size',
      help='Number of input records used per batch',
      default=1000,
      type=int)
  parser.add_argument(
      '--eval_batch_size',
      help='Number of eval records used per batch',
      default=500,
      type=int)
  parser.add_argument(
      '--train_steps', 
      help='Number of training steps to perform.', 
      default=1000,
      type=int)
  parser.add_argument(
      '--eval_steps',
      help='Number of evaluation steps to perform.',
      type=int,
      default=100)
  parser.add_argument(
      '--l2_regularization', 
      help='L2 Regularization', 
      type=int, 
      default=500)
  parser.add_argument(
      '--num_epochs', 
      help='Number of epochs', 
      default=5, 
      type=int)
  parser.add_argument(
      '--ignore_crosses',
      action='store_true',
      default=True,
      help='Whether to ignore crosses (linear model only).')
  return parser


def feature_columns(model_type, vocab_sizes, use_crosses):
  """Return the feature columns with their names and types."""
  result = []

  if model_type == LINEAR:
    
    # Base columns.
    for column_name in COLUMN_NAMES:
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      result.append(column)

    if use_crosses:
      
      # All pairs of columns we have added so far.
      for pair in itertools.combinations(result, 2):
        column = tf.contrib.layers.crossed_column(
            list(pair),
            hash_bucket_size=CROSS_HASH_BUCKET_SIZE,
            hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
            combiner='sum')
        result.append(column)

  elif model_type == DEEP:
    
    for column_name in COLUMN_NAMES:
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      embedding_size = int(math.floor(6 * vocab_size**0.25))
      embedding = tf.contrib.layers.embedding_column(column,
                                                     embedding_size,
                                                     combiner='mean')
      result.append(embedding)
  
  elif model_type == DEEP_CLASSIFIER:
    
    for column_name in COLUMN_NAMES:
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      embedding_size = int(math.floor(6 * vocab_size**0.25))
      embedding = tf.contrib.layers.embedding_column(column,
                                                     embedding_size,
                                                     combiner='mean')
      
      #if column_name == 'subreddit_id':
      #  column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(column_name, num_buckets=3 ))
      
      result.append(embedding)

  return result


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_transformed_reader_input_fn(transformed_metadata,transformed_data_paths,batch_size,mode):
  """Wrap the get input features function to provide the runtime arguments."""
  return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=(
          transformed_data_paths[0] if len(transformed_data_paths) == 1
          else transformed_data_paths),
      training_batch_size=batch_size,
      label_keys=[TARGET_FEATURE_COLUMN],
      reader=gzip_reader_fn,
      key_feature_name=KEY_FEATURE_COLUMN,
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
      num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def get_vocab_sizes():
  """Read vocabulary sizes from the metadata."""
  return {column_name: int(10*1000) for column_name in COLUMN_NAMES}


def model_fn_extra(estimator):
  '''
  A function that takes a specified estimator and returns a function
  that in turn returns an EstimatorSpec. When the estimatorSpec's 
  `export_outputs` is defined, it updates it to a PredictOutput created
  from the existing `predictions` dict.

  Intended to be passed as an arg to the `model_fn` arg in a 
  `tf.estimator.Estimator(model_fn=...)` call. 
  '''
  def _model_fn(features, labels, mode):
    estimatorSpec = estimator._call_model_fn(
      features=features, labels=labels, mode=mode, config=estimator.config)

    if estimatorSpec.export_outputs:
      # in serving mode, return the same fields as in prediction mode; this 
      # ensures that the instance key in 'predictions' will be among the 
      # outputs of the SavedModel SignatureDef.
      estimatorSpec.export_outputs['predict'] = tf.estimator.export.PredictOutput(estimatorSpec.predictions)

      estimatorSpec.export_outputs['serving_default'] = tf.estimator.export.PredictOutput(estimatorSpec.predictions)

    tf.logging.info('\nestimatorSpec prediction keys: {}\n'.format(estimatorSpec.predictions.keys()))

    return estimatorSpec
  return _model_fn
  

def get_experiment_fn(args):
  """Wrap the get experiment function to provide the runtime arguments."""
  vocab_sizes = get_vocab_sizes()
  use_crosses = not args.ignore_crosses

  def get_experiment(output_dir):
    """Function that creates an experiment http://goo.gl/HcKHlT.
    Args:
      output_dir: The directory where the training output should be written.
    Returns:
      A `tf.contrib.learn.Experiment`.
    """

    columns = feature_columns(args.model_type, vocab_sizes, use_crosses)
    runconfig = tf.contrib.learn.RunConfig()    
    cluster = runconfig.cluster_spec    
    num_table_shards = max(1, runconfig.num_ps_replicas * 3)
    num_partitions = max(1, 1 + cluster.num_tasks('worker') if cluster and 'worker' in cluster.jobs else 0)
    model_dir = os.path.join(output_dir, MODEL_DIR)
    
    if args.model_type == LINEAR:
      
      estimator = tf.contrib.learn.LinearRegressor(
          model_dir=model_dir,
          feature_columns=columns,
          optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
              example_id_column=KEY_FEATURE_COLUMN,
              symmetric_l2_regularization=args.l2_regularization,
              num_loss_partitions=num_partitions,
              num_table_shards=num_table_shards)
          )
    
    elif args.model_type == DEEP:
      
      estimator = tf.contrib.learn.DNNRegressor(
          hidden_units=args.hidden_units,
          feature_columns=columns,
          model_dir=model_dir)
    
    elif args.model_type == DEEP_CLASSIFIER:
      """
      # this works
      estimator = tf.contrib.learn.DNNClassifier(
          hidden_units=args.hidden_units,
          feature_columns=columns,
          model_dir=model_dir)
      """
      estimator_0 = tf.estimator.DNNClassifier(
          config=runconfig,
          hidden_units=args.hidden_units,
          feature_columns=columns,
          model_dir=model_dir)          
      estimator_1 = tf.contrib.estimator.forward_features(estimator_0, KEY_FEATURE_COLUMN)
      estimator = tf.estimator.Estimator(
                      model_fn=model_fn_extra(estimator_1),
                      config=runconfig
                      )
                        

    transformed_metadata = metadata_io.read_metadata(args.transformed_metadata_path)
    
    raw_metadata = metadata_io.read_metadata(args.raw_metadata_path)
    
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata,
            args.transform_savedmodel,
            raw_label_keys=[TARGET_FEATURE_COLUMN],
            #raw_feature_keys=[KEY_FEATURE_COLUMN],
            ))
    
    export_strategy = tf.contrib.learn.utils.make_export_strategy(
        serving_input_fn, 
        exports_to_keep=5,
        default_output_alternative_key=None
        )

    train_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, 
        args.train_data_paths, 
        args.batch_size,
        tf.contrib.learn.ModeKeys.TRAIN
        )

    eval_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, 
        args.eval_data_paths, 
        args.batch_size,
        tf.contrib.learn.ModeKeys.EVAL
        )

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        export_strategies=export_strategy,
        min_eval_frequency=500
        )

  # Return a function to create an Experiment.
  return get_experiment


def main(argv=None):
  """Run a Tensorflow model."""
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  learn_runner.run(experiment_fn=get_experiment_fn(args),
                   output_dir=output_dir
                   )


if __name__ == '__main__':
  main()