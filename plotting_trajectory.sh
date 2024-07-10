export CUDA_VISIBLE_DEVICES=0

ROOT=/ccvl/net/ccvl15/luoxin/coord/0709
OUTPUT_DIR=$ROOT/output
python plotting_trajectory.py \
    --root $ROOT \
    --output_dir $OUTPUT_DIR \
    --plot    