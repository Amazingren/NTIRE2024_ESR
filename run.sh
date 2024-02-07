CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir [path to your data dir] \
    --save_dir [path to your save dir] \
    --model_id 0

# # Evaluation on DIV2K valid set
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir [path to your data dir] \
#     --save_dir [path to your save dir] \
#     --model_id 0 \
#     --dataset_name DIV2K 

# # Evaluation on DIV2K valid & test sets
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir [path to your data dir] \
#     --save_dir [path to your save dir] \
#     --model_id 0 \
#     --dataset_name DIV2K --include_test

# # Evaluation on LSDIR valid set
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir [path to your data dir] \
#     --save_dir [path to your save dir] \
#     --model_id 0 \
#     --dataset_name LSDIR 

# # Evaluation on LSDIR valid & test sets
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir [path to your data dir] \
#     --save_dir [path to your save dir] \
#     --model_id 0 \
#     --dataset_name LSDIR --include_test

# # Evaluation on Hybrid test LR sets
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir [path to your data dir] \
#     --save_dir [path to your save dir] \
#     --model_id 0 \
#     --hybrid_test
