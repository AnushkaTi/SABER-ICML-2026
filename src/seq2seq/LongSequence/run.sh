# Use any number of tasks for task list
# Example Tasks: mnli cb wic copa

# Update label map - If new tasks
python update_label_map.py --tasks mnli cb wic copa

python train.py \
  --task_list mnli cb wic copa\
  --select_k_per_class 1000 \
  --batch_size 32 \
  --lr 0.3 \
  --num_epochs 10 \
  --freeze_weights \
  --prefix_len 10 \
  --model_name t5-large \
  --early_stopping \
  --test_eval_after_every_task \
