#!/bin/bash

python validate.py raft_small --pretrained_ckpt things \
--val_dataset sintel-clean-trainval+sintel-final-trainval+kitti-2015-trainval \
--output_path results/labeled_validations --write_outputs --flow_format flo --write_individual_metrics