# This config spends about 3 minutes to run on a Macmini with M4 Chip, 16GB RAM
uv run -m examples.bayes_inverse \
--method full_finetune \
--max_seq_len 256 \
--batch_size 8 \
--num_iterations 150 \
--learning_rate 1e-5 \
--save_checkpoint \
--checkpoint_path examples/ckpts/model_full_finetune.pt \
--dataset_path autograder/cpen455_released_datasets/augmented_train_val_subset.csv \
--early_stopping \
--patience 5 \
--min_delta 0.001 \
--eval_interval 10