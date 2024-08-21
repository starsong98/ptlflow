"""
AVAILABLE MODELS (alphabetic order):
* ccmr,ccmr+,craft,csflow,
* dicl,dip,
* fastflownet,flow1d,flowformer,flowformer++,flownet2,flownetc,flownetcs,flownetcss,flownets,flownetsd,
* gma,gmflow,gmflow+,gmflow+_sc2,gmflow+_sc2_refine6,gmflow_refine,gmflownet,gmflownet_mix,
* hd3,hd3_ctxt,
* irr_pwc,irr_pwcnet,irr_pwcnet_irr,
* lcv_raft,lcv_raft_small,
    * liteflownet,liteflownet2,liteflownet2_pseudoreg,liteflownet3,liteflownet3_pseudoreg,liteflownet3s,
    * liteflownet3s_pseudoreg,llaflow,llaflow_raft,
* maskflownet,maskflownet_s,matchflow,matchflow_raft,memflow,memflow_t,ms_raft+,
* neuflow,
* pwcnet,pwcnet_nodc,
* raft,raft_small,
    * rapidflow,rapidflow_it1,rapidflow_it12,rapidflow_it2,rapidflow_it3,rapidflow_it6,rpknet,
* scopeflow,scv4,scv8,
    * sea_raft,sea_raft_l,sea_raft_m,sea_raft_s,
    * separableflow,skflow,splatflow,starflow,
* unimatch,unimatch_sc2,unimatch_sc2_refine6,
* vcn,vcn_small,videoflow_bof,videoflow_mof

List of available checkpoints: https://ptlflow.readthedocs.io/en/latest/models/checkpoint_list.html
"""

import os

# all 15 test clips
test_clips = [
    #"datasets_realvideo/REDS4/train_blur/000",  # blurred HR
    #"datasets_realvideo/REDS4/train_blur/011",
    #"datasets_realvideo/REDS4/train_blur/015",
    #"datasets_realvideo/REDS4/train_blur/020",
    #"datasets_realvideo/REDS4/train_blur_bicubic/X4/000",  # blurred LR
    #"datasets_realvideo/REDS4/train_blur_bicubic/X4/011",
    #"datasets_realvideo/REDS4/train_blur_bicubic/X4/015",
    #"datasets_realvideo/REDS4/train_blur_bicubic/X4/020",
    #"datasets_realvideo/REDS4/train_sharp/000",  # sharp HR
    #"datasets_realvideo/REDS4/train_sharp/011",
    #"datasets_realvideo/REDS4/train_sharp/015",
    #"datasets_realvideo/REDS4/train_sharp/020",
    "datasets_realvideo/REDS4/train_sharp_bicubic/X4/000",  # sharp LR
    "datasets_realvideo/REDS4/train_sharp_bicubic/X4/011",
    "datasets_realvideo/REDS4/train_sharp_bicubic/X4/015",
    "datasets_realvideo/REDS4/train_sharp_bicubic/X4/020",
]

# all models you want to run
models = [
    ### SEA-RAFT
    "sea_raft_l",
    "sea_raft_m",
    "sea_raft_s",
    ### BASICS
    "raft",
    "gma",
    "skflow",
    #"flowformer",
    #"flowformer++"
    ### MULTI-SCALE
    "ms_raft+",
    #'ccmr',
    #'rpknet',
    ### MULTI-FRAME
    'videoflow_mof',
    'videoflow_bof',
    'memflow_t',
    'memflow',
    ### LIGHTWEIGHT
    #"rapidflow",
    #"neuflow",
]

# list of lists of checkpoints you want to run, for each model
checkpoints = [
    ["sintel", "spring",],  # sea_raft_l
    ["sintel", "spring",],  # sea_raft_m
    ["sintel", "spring",],  # sea_raft_s
    ["things", "sintel", "kitti"],    # raft
    ["things", "sintel", "kitti"],    # gma
    ["things", "sintel", "kitti"],    # skflow
    ["mixed"],   # MS-RAFT+
    #["sintel", "kitti"] # RAPID-Flow
    #["things", "sintel"],   # neuflow
    ['sintel', 'kitti'],    # videoflow-mof
    ['sintel', 'kitti'],    # videoflow-bof
    ['sintel', 'kitti',],    # memflow-T
    ['sintel', 'kitti', 'spring'],    # memflow

]

# output directory
#OUT_DIR = "results/REDS4_runs_auto_sharp"
#OUT_DIR = "results/REDS4_runs_auto_blurLRx4"
OUT_DIR = "results/REDS4_runs_auto_sharpLRx4"

# start iterating
for idx_model, model in enumerate(models):
    for checkpoint in checkpoints[idx_model]:
        for test_clip in test_clips:
            command = f"CUDA_VISIBLE_DEVICES=1 python infer.py {model} --input_path {test_clip}  \
                --write_outputs --output_path {OUT_DIR} --flow_format flo --fp16 --pretrained_ckpt {checkpoint}"
            #print(command)
            os.system(command)