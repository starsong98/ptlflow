import os

# all 15 test clips
test_clips = [
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST01_003_f0433",
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST02_045_f0465",
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST03_081_f4833",
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST04_140_f3889",
    "datasets_realvideo/XVFI/Longer_testset/Type1/TEST05_158_f0321",
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST06_001_f0273",
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST07_076_f1889",
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST08_079_f0321",
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST09_112_f0177",
    "datasets_realvideo/XVFI/Longer_testset/Type2/TEST10_172_f1905",
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST11_078_f4977",
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST12_087_f2721",
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST13_133_f4593",
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST14_146_f1761",
    "datasets_realvideo/XVFI/Longer_testset/Type3/TEST15_148_f0465",
]

# all models you want to run
models = [
    "irr_pwc"
]

# list of lists of checkpoints you want to run, for each model
checkpoints = [
    ["sintel", "kitti", "things"]
]

# output directory
OUT_DIR = "results/XVFI_OccEsts"

# start iterating
for idx_model, model in enumerate(models):
    for checkpoint in checkpoints[idx_model]:
        for test_clip in test_clips:
            command = f"CUDA_VISIBLE_DEVICES=1 python infer_irrocc.py {model} --input_path {test_clip}  \
                --write_outputs --output_path {OUT_DIR} --flow_format flo --fp16 --pretrained_ckpt {checkpoint}"
            #print(command)
            os.system(command)