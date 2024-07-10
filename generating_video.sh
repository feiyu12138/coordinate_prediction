ROOT=/ccvl/net/ccvl15/luoxin/coord/0709
IMAGE_DIR=/ccvl/net/ccvl15/luoxin/coord/0709/output
OUTPUT_PATH=/ccvl/net/ccvl15/luoxin/coord/0709/output/output.mp4

python generating_video.py \
    --root $ROOT \
    --image_dir $IMAGE_DIR \
    --output_path $OUTPUT_PATH
