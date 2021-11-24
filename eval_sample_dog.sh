#!/usr/bin/env bash

DATASET="Dog"
TRAIN_DIR="./$DATASET/SAC/TRAIN/Dog"
TEST_DIR="./$DATASET/SAC/TEST/Dog"

python eval_sample.py --checkpoint_path=$TRAIN_DIR \
                         --dataset_name=$DATASET \
                         --dataset_split_name='test' \
                         --dataset_dir="./$DATASET/Data/tfrecords" \
                         --eval_dir=$TEST_DIR \
                         --model_name='inception_v3_topk' \
                         --batch_size=16 \
                         --eval_image_size=448\
                         --gpus="1"\
                         --num_classes=200\
                         --feature_maps="Mixed_7c"\
                         --attention_maps="Mixed_7c"\
                         --num_parts=32\
			 --ignore_missing_vars=True
