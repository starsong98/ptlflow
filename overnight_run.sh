#!/bin/bash
#CUDA_VISIBLE_DEVICES=4 python infer.py sea_raft_l --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt spring
#CUDA_VISIBLE_DEVICES=4 python infer.py sea_raft_l --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt sintel
#CUDA_VISIBLE_DEVICES=4 python infer.py sea_raft_m --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt spring
#CUDA_VISIBLE_DEVICES=4 python infer.py sea_raft_m --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt sintel
#CUDA_VISIBLE_DEVICES=4 python infer.py sea_raft_s --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt spring
#CUDA_VISIBLE_DEVICES=4 python infer.py sea_raft_s --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt sintel

CUDA_VISIBLE_DEVICES=4 python infer.py ccmr \
--input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" \
--write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt sintel
CUDA_VISIBLE_DEVICES=4 python infer.py ccmr+ \
--input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" \
--write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt sintel
CUDA_VISIBLE_DEVICES=4 python infer.py rpknet \
--input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" \
--write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt spring
CUDA_VISIBLE_DEVICES=4 python infer.py rpknet \
--input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" \
--write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16 --pretrained_ckpt sintel
#CUDA_VISIBLE_DEVICES=4 python infer.py raft --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16
#CUDA_VISIBLE_DEVICES=4 python infer.py raft_small --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16
#CUDA_VISIBLE_DEVICES=4 python infer.py gmflow --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16
#CUDA_VISIBLE_DEVICES=4 python infer.py gmflow+ --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16
#CUDA_VISIBLE_DEVICES=4 python infer.py ccmr --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16
#CUDA_VISIBLE_DEVICES=4 python infer.py ccmr+ --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16
#CUDA_VISIBLE_DEVICES=4 python infer.py rpknet --input_path "/home/taewoosuh/ptlflow/datasets_realvideo/DAVIS/JPEGImages/Full-Resolution/breakdance" --write_outputs --output_path "/hdd/20245174/ptlflow_results/DAVIS_runs" --flow_format flo --fp16