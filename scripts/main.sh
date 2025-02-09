#!/bin/bash

DATA=./data
CFG=vit_b16  
SOURCE='none'
TARGET='all'

DATASET=$1
TRAINER=$2
CLUSTER=$3
GPU=$4



CUDA_VISIBLE_DEVICES=${GPU} \
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--source-domains ${SOURCE} \
--target-domains ${TARGET} \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir results/${DATASET}_${CLUSTER}/${TRAINER}/${SOURCE}_${TARGET} \
--eval-only \
TEST.PER_CLASS_RESULT False \
TRAINER.UNLABELED_CLUSTERS ${CLUSTER}
