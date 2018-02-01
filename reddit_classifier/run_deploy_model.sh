#!/bin/sh

PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
MODEL_NAME="redditcomments"
GCS_PATH="${BUCKET}/${MODEL_NAME}"
MODEL_VERSION="v3"
DEPLOYMENT_SOURCE="${GCS_PATH}/model/redditcomments_20180131_213022/model/export/Servo/1517434280"
REGION=us-central1

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "GCS_PATH="${GCS_PATH}
echo "MODEL_VERSION="${MODEL_VERSION}
echo "DEPLOYMENT_SOURCE="${DEPLOYMENT_SOURCE}
echo "--------------------------------------"


# create model if does not exist
gcloud ml-engine models create $MODEL_NAME \
    --regions $REGION

# create version
gcloud ml-engine versions create $MODEL_VERSION \
    --model $MODEL_NAME \
    --origin $DEPLOYMENT_SOURCE \
    --runtime-version=1.4

# set version to default
gcloud ml-engine versions set-default $MODEL_VERSION \
    --model=$MODEL_NAME

# desc model and version    
gcloud ml-engine versions describe $MODEL_VERSION \
    --model $MODEL_NAME

    