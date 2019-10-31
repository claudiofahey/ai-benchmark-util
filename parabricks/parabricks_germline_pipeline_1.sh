#!/usr/bin/env bash
# Sample bash script to run a single sample.
# This is not used by Slurm.
# Based on PARABRICKS_ISILON_NVIDIA_WP_v10072019.pdf.

set -ex

GENOME=NA12878
INPUT_DIR=/mnt/isilon1/data/genomics/from_nas/fq/${GENOME}
OUTPUT_DIR=/mnt/isilon1/data/genomics/output/${GENOME}
SCRATCH_DIR=/raid/genomics
REF_DIR=${SCRATCH_DIR}/reference_files
TMP_DIR=${SCRATCH_DIR}/tmp/${GENOME}

echo OUTPUT_DIR=${OUTPUT_DIR}
ls -lh ${INPUT_DIR}/*.fastq.gz
ls -lh \
    ${REF_DIR}/Homo_sapiens_assembly38.fasta \
    ${REF_DIR}/Homo_sapiens_assembly38.fasta.* \
    ${REF_DIR}/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz \
    ${REF_DIR}/Homo_sapiens_assembly38.dbsnp138.vcf

mkdir -p ${OUTPUT_DIR}
rm -rf ${TMP_DIR}
mkdir -p ${TMP_DIR}

_START_=`date`
echo "${GENOME} start = ${_START_}"
export NVIDIA_VISIBLE_DEVICES="0,1,2,3"

pbrun germline \
--ref ${REF_DIR}/Homo_sapiens_assembly38.fasta \
--in-fq ${INPUT_DIR}/*.fastq.gz \
"@RG\tID:foo0\tLB:lib1\tPL:bar\tSM:${GENOME}\tPU:unit0" \
--out-bam ${OUTPUT_DIR}/${GENOME}.bam \
--num-gpus 4 \
--out-recal-file ${OUTPUT_DIR}/${GENOME}.txt \
--knownSites ${REF_DIR}/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz \
--knownSites ${REF_DIR}/Homo_sapiens_assembly38.dbsnp138.vcf \
--out-variants ${OUTPUT_DIR}/${GENOME}.g.vcf.gz \
--gvcf \
--tmp-dir ${TMP_DIR}

pbrun deepvariant \
--ref ${REF_DIR}/Homo_sapiens_assembly38.fasta \
--num-gpus 4 \
--in-bam ${OUTPUT_DIR}/${GENOME}.bam \
--out-variants ${OUTPUT_DIR}/${GENOME}_dv.g.vcf.gz \
--gvcf

ls -lhR ${OUTPUT_DIR}

_END_=`date`
echo "${GENOME} end = ${_END_}"
