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

# all models you want to run
models = [
    ### SEA-RAFT
    #"sea_raft_l",
    #"sea_raft_m",
    #"sea_raft_s",
    ### BASICS
    #"raft",
    #"gma",
    #"skflow",
    #"gmflow",
    #"gmflow_refine",
    #"unimatch",
    #"pwcnet",
    #"irr_pwc",
    #"irr_pwcnet",
    #"dip",
    #"flowformer",
    #"flowformer++"
    ### MULTI-SCALE
    #"ms_raft+",
    #'ccmr',
    #'rpknet',
    ### MULTI-FRAME
    #'videoflow_mof',
    'videoflow_bof',
    #'memflow_t',
    #'memflow',
    "splatflow",    # this one requires cupy. okay.
    ### LIGHTWEIGHT
    #"rapidflow",
    #"neuflow",
    #"neuflow2"
]

# output directory
#OUT_DIR = "results/REDS4_runs_auto_sharp"
#OUT_DIR = "results/REDS4_runs_auto_blurLRx4"
#OUT_DIR = "results/REDS4_runs_auto_sharpLRx4"

# start iterating
for idx_model, model in enumerate(models):
    command = f"python validate.py {model} --pretrained_ckpt things \
--val_dataset sintel-clean-trainval+sintel-final-trainval+kitti-2015-trainval \
--output_path results/labeled_validations --write_outputs --flow_format original --write_individual_metrics"
    #print(command)
    os.system(command)