export CUDA_VISIBLE_DEVICES=0

ROOT=/media/luoxin/docs/24summer/chole_data/06282024_tissue_4/tissue_4
BATCH_SIZE=8
NUM_WORKERS=8
OUTPUT_DIR=$ROOT/output
BACKBONE=resnet
CKPT=/media/luoxin/docs/24summer/chole_data/model_final.pth
python predict_ood.py \
    --root $ROOT \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --n_locations 2 \
    --backbone $BACKBONE \
    --resume_path $CKPT \
    --plot
    