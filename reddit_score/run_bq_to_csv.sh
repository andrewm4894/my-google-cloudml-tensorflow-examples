
MODEL_NAME="redditscore"
PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
GCS_PATH="${BUCKET}/${MODEL_NAME}"
CSV_OUTPUT="${GCS_PATH}/data"

TRAIN_DATA="fh-bigquery.reddit_comments.2015_12"
EVAL_DATA="fh-bigquery.reddit_comments.2016_01"
PREDICT_DATA="fh-bigquery.reddit_comments.2016_02"

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "GCS_PATH="${GCS_PATH}
echo "CSV_OUTPUT="${CSV_OUTPUT}
echo "TRAIN_DATA="${TRAIN_DATA}
echo "EVAL_DATA="${EVAL_DATA}
echo "PREDICT_DATA="${PREDICT_DATA}
echo "--------------------------------------"

python2 bq_to_csv.py --training_data "${TRAIN_DATA}" \
                     --eval_data "${EVAL_DATA}" \
                     --predict_data "${PREDICT_DATA}" \
                     --output_dir "${CSV_OUTPUT}" \
                     --project_id "${PROJECT}" \
                     --model_name "${MODEL_NAME}"