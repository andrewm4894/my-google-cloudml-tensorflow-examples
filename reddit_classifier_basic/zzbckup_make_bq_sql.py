
"""Schema and tranform definition for the Reddit dataset."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import coders
from tensorflow_transform.tf_metadata import dataset_schema


def make_standard_sql(table_name,
                      mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Returns Standard SQL string to populate reddit features.
  This query takes ~60s, processing 13.3GB for reddit_comments.2015_12 table,
  and takes ~140s, processing 148GB for reddit_comments.2015_* table.
  Args:
    table_name: the table name to pull the data from.
      multiple tables can be chosen using *, like:
      fh-bigquery.reddit_comments.2015_*
    mode: if the mode is INFER, 'score' field is not populated.
  Returns:
    The standard SQL query to pull the features from the given reddit table.
  """

  infer_mode = (mode == tf.contrib.learn.ModeKeys.INFER)
  return """
SELECT
  {score_optional}
  created_utc,
  COALESCE(subreddit, '') AS subreddit,
  COALESCE(author, '') AS author,
  COALESCE(REGEXP_REPLACE(body, r'\\n+', ' '), '') AS comment_body,
  '' AS comment_parent_body,
  1 AS toplevel,
  id as example_id
FROM
  `{table_name}`
WHERE
  (score_hidden IS NULL OR score_hidden = false) 
  AND
  SUBSTR(parent_id, 1, 2) = 't3' 
  AND
  # limit to subs of interest
  subreddit in ('news','pics','ireland')
  
# extra bit of shuffling when the limit condition is in place
ORDER BY RAND() 

# limit to a certain amount of records for developing on
LIMIT 10000 

""".format(
    score_optional=('' if infer_mode else 'if(score>0,1,0) as score,'),
    table_name=table_name)


def make_csv_coder(schema, mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Produces a CsvCoder from a data schema.
  Args:
    schema: A tf.Transform `Schema` object.
    mode: tf.contrib.learn.ModeKeys specifying if the source is being used for
      train/eval or prediction.
  Returns:
    A tf.Transform CsvCoder.
  """
  column_names = [] if mode == tf.contrib.learn.ModeKeys.INFER else ['score']
  column_names += [
      'created_utc', 'subreddit', 'author', 'comment_body',
      'comment_parent_body', 'toplevel', 'example_id'
  ]
  return coders.CsvCoder(column_names, schema)


def make_input_schema(mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Input schema definition.
  Args:
    mode: tf.contrib.learn.ModeKeys specifying if the schema is being used for
      train/eval or prediction.
  Returns:
    A `Schema` object.
  """
  result = ({} if mode == tf.contrib.learn.ModeKeys.INFER else {
      'score': tf.FixedLenFeature(shape=[], dtype=tf.int64)
  })
  result.update({
      'subreddit': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'author': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'comment_body': tf.FixedLenFeature(shape=[], dtype=tf.string,
                                         default_value=''),
      'comment_parent_body': tf.FixedLenFeature(shape=[], dtype=tf.string,
                                                default_value=''),
      'toplevel': tf.FixedLenFeature(shape=[], dtype=tf.int64),
      'example_id': tf.FixedLenFeature(shape=[], dtype=tf.string),
  })
  return dataset_schema.from_feature_spec(result)


def make_preprocessing_fn(frequency_threshold):
  """Creates a preprocessing function for reddit.
  Args:
    frequency_threshold: The frequency_threshold used when generating
      vocabularies for categorical and text features.
  Returns:
    A preprocessing function.
  """

  def preprocessing_fn(inputs):
    """User defined preprocessing function for reddit columns.
    Args:
      inputs: dictionary of input `tensorflow_transform.Column`.
    Returns:
      A dictionary of `tensorflow_transform.Column` representing the transformed
          columns.
    """
    # TODO(b/35001605) Make this "passthrough" more DRY.
    result = {'score': inputs['score'], 'toplevel': inputs['toplevel']}

    result['subreddit_id'] = tft.string_to_int(
        inputs['subreddit'], frequency_threshold=frequency_threshold)

    for name in ('author', 'comment_body', 'comment_parent_body'):
      words = tf.string_split(inputs[name])
      # TODO(b/33467613) Translate these to bag-of-words style sparse features.
      result[name + '_bow'] = tft.string_to_int(
          words, frequency_threshold=frequency_threshold)

    return result

  return preprocessing_fn