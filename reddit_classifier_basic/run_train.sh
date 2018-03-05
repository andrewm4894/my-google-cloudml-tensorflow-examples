
PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
MODEL_NAME="redditcomments"
GCS_PATH="${BUCKET}/${MODEL_NAME}"
PREPROCESS_OUTPUT="${GCS_PATH}/data/20180305_221837"
JOB_ID="${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${GCS_PATH}/model/${JOB_ID}"
STAGING_BUCKET="gs://zz-pmc-ml-staging-${JOB_ID}"
REGION="us-central1"

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "GCS_PATH="${GCS_PATH}
echo "PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT}
echo "JOB_ID="${JOB_ID}
echo "OUTPUT_PATH="${OUTPUT_PATH}
echo "--------------------------------------"

# make staging bucket
gsutil mb -l $REGION ${STAGING_BUCKET}

# run job
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$STAGING_BUCKET" \
  --region us-central1 \
  --config config.yaml \
  -- \
  --model_type deep_classifier \
  --hidden_units 512 512 \
  --batch_size 512 \
  --train_steps 500 \
  --eval_steps 250 \
  --output_path "${GCS_PATH}/model/${JOB_ID}" \
  --raw_metadata_path "${PREPROCESS_OUTPUT}/raw_metadata" \
  --transformed_metadata_path "${PREPROCESS_OUTPUT}/transformed_metadata" \
  --transform_savedmodel "${PREPROCESS_OUTPUT}/transform_fn" \
  --eval_data_paths "${PREPROCESS_OUTPUT}/features_eval*" \
  --train_data_paths "${PREPROCESS_OUTPUT}/features_train*"
