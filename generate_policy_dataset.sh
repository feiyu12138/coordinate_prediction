ROOT=/ccvl/net/ccvl15/luoxin/coord/0709/output
INFO=$ROOT/coords.json
OUT=$ROOT/policy
python generate_policy_dataset.py \
    --info_path $INFO \
    --output_path $OUT 