export CUDA_VISIBLE_DEVICES=0

ROOT=/ccvl/net/ccvl15/luoxin/coord/0709
BATCH_SIZE=32
NUM_WORKERS=32
OUTPUT_DIR=$ROOT/output
BACKBONE=resnet
CKPT=/ccvl/net/ccvl15/luoxin/coord/0709/ckpt/model_9.pth
python predict.py \
    --root $ROOT \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --n_locations 2 \
    --backbone $BACKBONE \
    --resume_path $CKPT
    