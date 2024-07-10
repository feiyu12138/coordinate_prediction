export CUDA_VISIBLE_DEVICES=0

ROOT=/media/luoxin/docs/24summer/chole_data/06282024_tissue_4/tissue_4/
OUTPUT_DIR=$ROOT/output
python plotting_trajectory_segment.py \
    --root $ROOT \
    --output_dir $OUTPUT_DIR \
    --plot    