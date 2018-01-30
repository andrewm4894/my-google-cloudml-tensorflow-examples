
MODEL_NAME="redditcomments"
PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
GCS_PATH="${BUCKET}/${MODEL_NAME}"
PREPROCESS_OUTPUT="${GCS_PATH}/$(date +%Y%m%d_%H%M%S)"

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "GCS_PATH="${GCS_PATH}
echo "PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT}
echo "--------------------------------------"

python2 preprocess.py --training_data fh-bigquery.reddit_comments.2015_12 \
                     --eval_data fh-bigquery.reddit_comments.2016_01 \
                     --predict_data fh-bigquery.reddit_comments.2016_02 \
                     --output_dir "${PREPROCESS_OUTPUT}" \
                     --project_id "${PROJECT}" \
                     --cloud \
                     --model_name "${MODEL_NAME}"