import os

# all 15 test clips
test_clips = [
    "datasets_realvideo/Reflective_scenes/marblefloor"
]

# all models you want to run
models = [
    #"sea_raft_l",
    #"sea_raft_m",
    #"sea_raft_s",
    "raft",
    #"gma"  # Out of Memory
    #"neuflow",
    #"ms_raft+",
    #"rapidflow",
]

# list of lists of checkpoints you want to run, for each model
checkpoints = [
    #["sintel", "spring",],  # sea_raft_l
    #["sintel", "spring",],  # sea_raft_m
    #["sintel", "spring",],  # sea_raft_s
    #["sintel", "kitti"],    # raft
    ["sintel"],    # raft
    #["sintel", "kitti"],    # gma
    #["things", "sintel"],   # neuflow
    #["mixed"],   # MS-RAFT+
    #["sintel", "kitti"] # RAPID-Flow
]

# output directory
OUT_DIR = "results/Reflector_runs"

# start iterating
for idx_model, model in enumerate(models):
    for checkpoint in checkpoints[idx_model]:
        for test_clip in test_clips:
            command = f"CUDA_VISIBLE_DEVICES=1 python infer.py {model} --input_path {test_clip}  \
                --write_outputs --output_path {OUT_DIR} --flow_format flo --fp16 --pretrained_ckpt {checkpoint}"
            #print(command)
            os.system(command)