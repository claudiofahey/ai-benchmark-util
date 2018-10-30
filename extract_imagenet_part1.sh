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
VALIDATION_TARBALL="${DATA_DIR}/ILSVRC2012_img_val.tar"
TRAIN_TARBALL="${DATA_DIR}/ILSVRC2012_img_train.tar"

[ -f "${BBOX_TAR_BALL}" ] || { echo "File ${BBOX_TAR_BALL} not found" ; exit 1 ;}
[ -f "${VALIDATION_TARBALL}" ] || { echo "File ${VALIDATION_TARBALL} not found" ; exit 1 ;}
[ -f "${TRAIN_TARBALL}" ] || { echo "File ${TRAIN_TARBALL} not found" ; exit 1 ;}

mkdir -p "${BBOX_DIR}"
tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}"
"${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" | sort > "${BOUNDING_BOX_FILE}"

# Extract all images from the ImageNet 2012 validation dataset.
OUTPUT_PATH="${SCRATCH_DIR}/validation/"
mkdir -p "${OUTPUT_PATH}"
tar xf "${VALIDATION_TARBALL}" -C "${OUTPUT_PATH}"

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
VALIDATION_DIRECTORY="${SCRATCH_DIR}/validation/"
PREPROCESS_VAL_SCRIPT="./preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="imagenet_2012_validation_synset_labels.txt"
"${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

# Extract all tar files from the ImageNet 2012 train dataset.
OUTPUT_PATH="${SCRATCH_DIR}/train/"
mkdir -p "${OUTPUT_PATH}"
tar xvf "${TRAIN_TARBALL}" -C "${OUTPUT_PATH}"
