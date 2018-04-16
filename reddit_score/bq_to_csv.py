
"""Pull data from BigQuery and save to csv files on GCS."""

import argparse
import random
import subprocess
import sys

import apache_beam as beam
import datetime, os


def _default_project():
    get_project = [
      'gcloud', 'config', 'list', 'project', '--format=value(core.project)'
    ]

    with open(os.devnull, 'w') as dev_null:
        return subprocess.check_output(get_project, stderr=dev_null).strip()


def parse_arguments(argv):
    """Parse command line arguments.
    Args:
      argv: list of command line arguments including program name.
    Returns:
      The parsed arguments as returned by argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description='Runs Preprocessing on the model data.'
        )
    parser.add_argument(
        '--project_id', 
        help='The project to which the job will be submitted.'
        )
    parser.add_argument(
        '--model_name', 
        help='The name for you model to appear in jobs etc.',
        default='mymodel'
        )
    parser.add_argument(
        '--training_data',
        required=True,
        help='Data to analyze and encode as training features.'
        )
    parser.add_argument(
        '--eval_data',
        required=True,
        help='Data to encode as evaluation features.'
        )
    parser.add_argument(
        '--predict_data', 
        help='Data to encode as prediction features.'
        )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Google Cloud Storage or Local directory in which to place outputs.'
        )
    args, _ = parser.parse_known_args(args=argv[1:])

    if not args.project_id:
        args.project_id = _default_project()

    return args


def to_csv(rowdict):
    # Pull columns from BQ and create a line
    import hashlib
    import copy

    #CSV_COLUMNS = 'example_id,subreddit,comment,score,comment_ints'.split(',')
    CSV_COLUMNS = 'example_id,subreddit,comment,comment_ints,score'.split(',')

    for result in [rowdict]:
        data = '|'.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
        yield str('{}'.format(data))


def vocab_to_csv(rowdict):
    # Pull columns from BQ and create a line
    import hashlib
    import copy

    CSV_COLUMNS = 'word'.split(',')

    for result in [rowdict]:
        data = ','.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
        yield str('{}'.format(data))

  
def beam_bq_to_csv():
  
    args = parse_arguments(sys.argv)

    import shutil, os, subprocess
    job_name = '{}-bq-to-csv-{}'.format(args.model_name,datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

    print 'Launching Dataflow job {} ...'.format(job_name)

    OUTPUT_DIR = args.output_dir

    # if there is anything already in the target folder clear it out
    try:
        subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
    except:
        pass

    # define options for beam job
    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': job_name,
        'project': args.project_id,
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    }
    
    opts = beam.pipeline.PipelineOptions(flags = [], **options)

    # define runner for beam pipeline
    RUNNER = 'DataflowRunner'
        
    # create beam pipeline
    p = beam.Pipeline(RUNNER, options = opts)

    # bql to be run
    base_query = """

    SELECT 
      example_id,
      subreddit,
      ifnull(regexp_extract(comment,r'^((?:\\S+\\s+){{1}}\\S+).*'),'EMPTY EMPTY') as comment,
      concat("[" , 
        cast(cast(floor(rand()*100) as INT64) as STRING), "," , 
        cast(cast(floor(rand()*100) as INT64) as STRING), 
        "]" ) as comment_ints,
      score
    FROM
      (
      SELECT
        id AS example_id,
        score,
        subreddit,
        REPLACE(REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(lower(body), r'[^\\x00-\\x7F]', ''),r'\\n',' '),',',' '),r'[^a-zA-Z0-9. ]',''),'.',' . '),'  ',' ') AS comment, # remove non ascii chars and just leave english letters and numbers
        rand() AS rand_num
      FROM
        `{table_name}`
      WHERE
        (score_hidden IS NULL OR score_hidden = false) 
        AND
        # limit to subs of interest
        subreddit IN ('news','pics','ireland')
        AND
        body not IN ('[Deleted]','[Removed]')
      ORDER BY rand_num
      )
    LIMIT 10000

    """ 

    vocab_query = """

    WITH 

    base_data AS 
      (
        {base_data}
      ),
      
    words_data AS 
      (
        SELECT 
          example_id, 
          SPLIT(comment,' ') AS words
        FROM 
          base_data
      ),
      
    word_occurances AS 
      (
        SELECT 
          example_id,
          words
        FROM 
          words_data
          CROSS JOIN UNNEST(words_data.words) AS words
        GROUP BY 1,2
      ),
      
    wc_data AS
      (
      SELECT 
        APPROX_TOP_COUNT(words, 10000) AS wc 
      FROM 
        word_occurances
      )
     
    SELECT 'xyzpadxyz' AS word # add pad word
    UNION ALL
    SELECT 
      value AS word 
      --count
    FROM
      wc_data, 
      unnest(wc_data.wc)
    UNION ALL
    SELECT '<UNK>' AS word # add unknown word

    """

    # export each training set
    for step in ['train', 'eval', 'predict', 'vocab']:

        # define sql to select data
        if step == 'vocab':
            
            selquery = vocab_query.format(
              base_data=base_query.format(
                table_name=args.training_data)
              )

        elif step == 'train':
            
            selquery = base_query.format(table_name=args.training_data)
        
        elif step == 'eval':
            
            selquery = base_query.format(table_name=args.eval_data)
        
        else:
            
            selquery = base_query.format(table_name=args.predict_data)

        # define pipeline
        if step == 'vocab':

            (p 
             | '{}_read'.format(step) >> beam.io.Read(beam.io.BigQuerySource(query = selquery, use_standard_sql = True))
             | '{}_csv'.format(step) >> beam.FlatMap(vocab_to_csv)
             | '{}_out'.format(step) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{}.csv'.format(step)), num_shards=1))
            )

        else:

            (p 
             | '{}_read'.format(step) >> beam.io.Read(beam.io.BigQuerySource(query = selquery, use_standard_sql = True))
             | '{}_csv'.format(step) >> beam.FlatMap(to_csv)
             | '{}_out'.format(step) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{}.csv'.format(step))))
            )

    # kick off the beam job
    job = p.run()
    
# execute function to run the job
beam_bq_to_csv()