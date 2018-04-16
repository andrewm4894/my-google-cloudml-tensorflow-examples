
PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
MODEL_NAME="redditscore"
JOB_ID="${MODEL_NAME}_train_$(date +%Y%m%d_%H%M%S)"
REGION="us-central1"
STAGING_BUCKET="gs://pmc-ml-staging"
INPUT_PATH="${BUCKET}/${MODEL_NAME}/data"
OUTPUT_PATH="${BUCKET}/${MODEL_NAME}/trained_model"

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "JOB_ID="${JOB_ID}
echo "INPUT_PATH="${INPUT_PATH}
echo "OUTPUT_PATH="${OUTPUT_PATH}
echo "STAGING_BUCKET="${STAGING_BUCKET}
echo "REGION="${REGION}
echo "--------------------------------------"

# clear output path to start fresh each time
gsutil -m rm -r ${OUTPUT_PATH}

# launch train job
gcloud ml-engine jobs submit training ${JOB_ID} \
  --region=${REGION} \
  --module-name=trainer.task \
  --package-path=$(pwd)/trainer \
  --job-dir=${OUTPUT_PATH} \
  --staging-bucket=${STAGING_BUCKET} \
  --scale-tier=BASIC \
  --runtime-version=1.4 \
  -- \
  --bucket=${BUCKET} \
  --input_dir=${INPUT_PATH} \
  --output_dir=${OUTPUT_PATH} \
  --train_examples=10000

# PROJECT="pmc-analytical-data-mart"
# BUCKET="gs://pmc-ml-samples"
# MODEL_NAME="redditscore"
# GCS_PATH="${BUCKET}/${MODEL_NAME}"
# PREPROCESS_OUTPUT="${GCS_PATH}/data/20180306_092722"
# JOB_ID="${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
# OUTPUT_PATH="${GCS_PATH}/model/${JOB_ID}"
# STAGING_BUCKET="gs://zz-pmc-ml-staging-${JOB_ID}"
# REGION="us-central1"

# echo "--------------------------------------"
# echo "PROJECT="${PROJECT}
# echo "BUCKET="${BUCKET}
# echo "MODEL_NAME="${MODEL_NAME}
# echo "GCS_PATH="${GCS_PATH}
# echo "PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT}
# echo "JOB_ID="${JOB_ID}
# echo "OUTPUT_PATH="${OUTPUT_PATH}
# echo "--------------------------------------"

# # make staging bucket
# gsutil mb -l $REGION ${STAGING_BUCKET}

# # run job
# gcloud ml-engine jobs submit training "$JOB_ID" \
#   --module-name trainer.task \
#   --package-path trainer \
#   --staging-bucket "$STAGING_BUCKET" \
#   --region us-central1 \
#   --config config.yaml \
#   -- \
#   --model_type deep_classifier \
#   --hidden_units 512 512 \
#   --batch_size 512 \
#   --train_steps 500 \
#   --eval_steps 250 \
#   --output_path "${GCS_PATH}/model/${JOB_ID}" \
#   --raw_metadata_path "${PREPROCESS_OUTPUT}/raw_metadata" \
#   --transformed_metadata_path "${PREPROCESS_OUTPUT}/transformed_metadata" \
#   --transform_savedmodel "${PREPROCESS_OUTPUT}/transform_fn" \
#   --eval_data_paths "${PREPROCESS_OUTPUT}/features_eval*" \
#   --train_data_paths "${PREPROCESS_OUTPUT}/features_train*"
