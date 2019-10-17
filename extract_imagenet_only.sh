#!/bin/bash
#
# This script performs all steps required to build the TFRecord files from the ImageNet tar files.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

set -ex

DATA_DIR="/imagenet-data"
SCRATCH_DIR="/imagenet-scratch"
SYNSETS_FILE="imagenet_lsvrc_2015_synsets.txt"
BBOX_TAR_BALL="${DATA_DIR}/ILSVRC2012_bbox_train_v2.tar.gz"
VALIDATION_TARBALL="${DATA_DIR}/ILSVRC2012_img_val.tar"
VAL_LABELS_FILE="imagenet_2012_validation_synset_labels.txt"
TRAIN_TARBALL="${DATA_DIR}/ILSVRC2012_img_train.tar"
IMAGENET_METADATA_FILE="imagenet_metadata.txt"
BOUNDING_BOX_SCRIPT="./process_bounding_boxes.py"
PREPROCESS_VAL_SCRIPT="./preprocess_imagenet_validation_data.py"
BUILD_SCRIPT="./build_imagenet_data.py"
BBOX_DIR="${SCRATCH_DIR}/bounding_boxes"
BOUNDING_BOX_FILE="${SCRATCH_DIR}/imagenet_2012_bounding_boxes.csv"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/validation/"
TRAIN_DIRECTORY="${SCRATCH_DIR}/train/"
OUTPUT_DIRECTORY="${SCRATCH_DIR}/tfrecords2"

[ -d "${DATA_DIR}" ] || { echo "Directory ${DATA_DIR} not found" ; exit 1 ;}
[ -d "${SCRATCH_DIR}" ] || { echo "Directory ${SCRATCH_DIR} not found" ; exit 1 ;}
[ -f "${SYNSETS_FILE}" ] || { echo "File ${SYNSETS_FILE} not found" ; exit 1 ;}
[ -f "${BBOX_TAR_BALL}" ] || { echo "File ${BBOX_TAR_BALL} not found" ; exit 1 ;}
[ -f "${BOUNDING_BOX_SCRIPT}" ] || { echo "File ${BOUNDING_BOX_SCRIPT} not found" ; exit 1 ;}
[ -f "${PREPROCESS_VAL_SCRIPT}" ] || { echo "File ${PREPROCESS_VAL_SCRIPT} not found" ; exit 1 ;}
[ -f "${BUILD_SCRIPT}" ] || { echo "File ${BUILD_SCRIPT} not found" ; exit 1 ;}
[ -f "${VAL_LABELS_FILE}" ] || { echo "File ${VAL_LABELS_FILE} not found" ; exit 1 ;}
[ -f "${VALIDATION_TARBALL}" ] || { echo "File ${VALIDATION_TARBALL} not found" ; exit 1 ;}
[ -f "${TRAIN_TARBALL}" ] || { echo "File ${TRAIN_TARBALL} not found" ; exit 1 ;}
[ -f "${IMAGENET_METADATA_FILE}" ] || { echo "File ${IMAGENET_METADATA_FILE} not found" ; exit 1 ;}

# Validate source files.
#(cd "${DATA_DIR}" && md5sum -c ilsvrc2012.md5)

# Extract all images from the ImageNet 2012 validation dataset.
mkdir -p "${VALIDATION_DIRECTORY}"
tar xf "${VALIDATION_TARBALL}" -C "${VALIDATION_DIRECTORY}"

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
"${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

# Extract all tar files from the ImageNet 2012 train dataset.
mkdir -p "${TRAIN_DIRECTORY}"
tar xvf "${TRAIN_TARBALL}" -C "${TRAIN_DIRECTORY}"

# Extract the training files nXXXXXXXX.tar to nXXXXXXXX/*.JPG.
mpirun --allow-run-as-root -np 8 -H localhost:8 ./extract_imagenet_part2_mpi.py -i "${TRAIN_DIRECTORY}" -o "${TRAIN_DIRECTORY}"
rm -rf "${TRAIN_DIRECTORY}"*.tar

