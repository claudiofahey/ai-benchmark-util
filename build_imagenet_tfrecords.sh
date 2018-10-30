#!/bin/bash
set -ex

DATA_DIR="${1:-/imagenet-data}"
SYNSETS_FILE="${2:-./imagenet_lsvrc_2015_synsets.txt}"
SCRATCH_DIR="/imagenet-scratch"

BBOX_DIR="${SCRATCH_DIR}/bounding_boxes"
BBOX_TAR_BALL="${DATA_DIR}/ILSVRC2012_bbox_train_v2.tar.gz"
BOUNDING_BOX_SCRIPT="./process_bounding_boxes.py"
BOUNDING_BOX_FILE="${SCRATCH_DIR}/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${SCRATCH_DIR}/bounding_boxes/"
LABELS_FILE="${SYNSETS_FILE}"
TRAIN_DIRECTORY="${SCRATCH_DIR}/train/"
VALIDATION_TARBALL="${DATA_DIR}/ILSVRC2012_img_val.tar"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/validation/"
TRAIN_TARBALL="${DATA_DIR}/ILSVRC2012_img_train.tar"
BUILD_SCRIPT="./build_imagenet_data.py"
OUTPUT_DIRECTORY="${SCRATCH_DIR}/tfrecords"
IMAGENET_METADATA_FILE="./imagenet_metadata.txt"

mkdir -p "${OUTPUT_DIRECTORY}"

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"
