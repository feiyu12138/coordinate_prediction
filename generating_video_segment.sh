ROOT=/media/luoxin/docs/24summer/chole_data/06282024_tissue_4/tissue_4
IMAGE_DIR=/media/luoxin/docs/24summer/chole_data/06282024_tissue_4/tissue_4/output
OUTPUT_PATH=/media/luoxin/docs/24summer/chole_data/06282024_tissue_4/tissue_4/output/output.mp4
INFO_PATH=/media/luoxin/docs/24summer/chole_data/06282024_tissue_4.txt
python generating_video_segment.py \
    --root $ROOT \
    --image_dir $IMAGE_DIR \
    --output_path $OUTPUT_PATH \
    --info_path $INFO_PATH
