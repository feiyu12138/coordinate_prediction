export CUDA_VISIBLE_DEVICES=0

export WANDB_API_KEY='46e587ae4112a04da96b68ba807395204be787c9'
export WANDB_PROJECT='coord'
export WANDB_ENTITY='mid_level'

ROOT=/ccvl/net/ccvl15/luoxin/coord/0709
EPOCH=10
LR=2.5e-4
BATCH_SIZE=32
NUM_WORKERS=32
SAVE_DIR=$ROOT/ckpt
BACKBONE=resnet
EVAL_INTERVAL=1

python train.py \
    --root $ROOT \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --n_epochs $EPOCH \
    --save_dir $SAVE_DIR \
    --n_locations 2 \
    --backbone $BACKBONE \
    --do_train \
    --do_eval \
    --eval_interval $EVAL_INTERVAL \
    --project $WANDB_PROJECT \
    --entity $WANDB_ENTITY 
