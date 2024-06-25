# --- Evaluation on LSDIR_DIV2K_valid datasets for One Method:
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir [path to your data dir] \
    --save_dir [path to your save dir] \
    --model_id 0

# --- When Test datasets are included (For organizer)
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir ../ \
#     --save_dir ../results \
#     --include_test \
#     --model_id 0


# --- Test all the methods
#!/bin/bash
# DATA_DIR="/Your/Validate/Datasets/Path"
# SAVE_DIR="./results"
# MODEL_IDS=(
#     0 1 3 4 5 7 10 11 13 15 
#     16 17 18 19 21 23 25 26 
#     28 29 30 31 33 34 38 39 
#     41 42 43 44 45 46 48
# )

# for model_id in "${MODEL_IDS[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" --include_test --model_id "$model_id"
# done
