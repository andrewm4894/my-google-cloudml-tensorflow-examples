#!/bin/sh

set -e

PROJECT="pmc-analytical-data-mart"
BUCKET="gs://pmc-ml-samples"
MODEL_NAME="redditcomments"
GCS_PATH="${BUCKET}/${MODEL_NAME}"
MODEL_VERSION="v1"
SOURCE="${GCS_PATH}/predictions/20180206_163511"
TARGET_TABLE="zz_tmp_will_expire.${MODEL_NAME}_${MODEL_VERSION}_predictions_test"
  
printf '\n##################################\n'
printf 'DISPLAY INPUTS'
printf '\n##################################\n'

echo "--------------------------------------"
echo "PROJECT="${PROJECT}
echo "BUCKET="${BUCKET}
echo "MODEL_NAME="${MODEL_NAME}
echo "GCS_PATH="${GCS_PATH}
echo "SOURCE="${SOURCE}
echo "TARGET_TABLE="${TARGET_TABLE}
echo "--------------------------------------"

printf '\n##################################\n'
printf 'LOAD PREDICTIONS'
printf '\n##################################\n'

# load predictions
bq load --source_format=NEWLINE_DELIMITED_JSON --autodetect --replace \
    $TARGET_TABLE \
    $SOURCE/prediction.results*

# view predictions table
bq show $TARGET_TABLE
