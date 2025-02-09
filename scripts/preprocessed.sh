#!/bin/bash

DATA=./data
CFG=vit_b16  
SOURCE='none'
TARGET='all'

DATASET=$1
TRAINER=$2
CLUSTER=$3
GPU=$4
LOADER=$5   # 'trainu' or 'val'



CUDA_VISIBLE_DEVICES=${GPU} \
python preprocess_cluster.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--source-domains ${SOURCE} \
--target-domains ${TARGET} \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir results/${DATASET}_${CLUSTER}/${TRAINER}/${SOURCE}_${TARGET} \
--eval-only \
TRAINER.UNLABELED_CLUSTERS ${CLUSTER} \
DATASET.CLUSTER_LOADER ${LOADER}
