#!/bin/sh

PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
MODEL_NAME="redditscore"
GCS_PATH="${BUCKET}/${MODEL_NAME}"
PREPROCESS_OUTPUT="${GCS_PATH}/data/20180305_214219"

today=$(date --date="-0 days" +%Y%m%d)
now=$(date +"%Y%m%d_%H%M%S")

JOB_NAME="${MODEL_NAME}_predict_${now}"
REGION="us-central1"
DATA_FORMAT="TEXT"
INPUT_PATHS="${BUCKET}/${MODEL_NAME}/data/predict*"
OUTPUT_PATH="${BUCKET}/${MODEL_NAME}/predictions/${now}"

printf '\n##################################\n'
printf 'DISPLAY INPUTS'
printf '\n##################################\n'

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "JOB_NAME="${JOB_NAME}
echo "INPUT_PATHS="${INPUT_PATHS}
echo "OUTPUT_PATH="${OUTPUT_PATH}
echo "--------------------------------------"

printf '\n##################################\n'
printf 'RUN ONLINE PREDICTION TEST'
printf '\n##################################\n'

python2 predict_online_test.py

printf '\n##################################\n'
printf 'RUN BATCH PREDICTION'
printf '\n##################################\n'

# run batch prediction on gcs test data     
gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model $MODEL_NAME \
    --input-paths $INPUT_PATHS \
    --output-path $OUTPUT_PATH \
    --region $REGION \
    --data-format $DATA_FORMAT    

# capture status of last command and exit if error
status=$?
if [ $status -ne 0 ]; then
  echo "Return code was not zero but $status"
  exit $status  
fi      

gcloud ml-engine jobs describe $JOB_NAME
