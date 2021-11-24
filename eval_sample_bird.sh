#!/usr/bin/env bash

DATASET="Bird"
WEIGHT_DIR="./$DATASET/SAC/TRAIN/Bird"
TEST_DIR="./$DATASET/SAC/TEST/Bird"

python eval_sample.py --checkpoint_path=$WEIGHT_DIR \
                         --dataset_name=$DATASET \
                         --dataset_split_name='test' \
                         --dataset_dir="./$DATASET/Data/tfrecords" \
                         --eval_dir=$TEST_DIR \
                         --model_name='inception_v3_topk' \
                         --batch_size=16 \
                         --eval_image_size=448\
                         --gpus="0"\
                         --num_classes=200\
                         --feature_maps="Mixed_6e"\
                         --attention_maps="Mixed_7a_b0"\
                         --num_parts=32
