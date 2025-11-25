# This config spends about 3 minutes to run on a Macmini with M4 Chip, 16GB RAM
uv run -m examples.bayes_inverse \
--method full_finetune \
--max_seq_len 256 \
--batch_size 8 \
--num_iterations 150 \
--learning_rate 1e-5 \
--dataset_path autograder/cpen455_released_datasets/augmented_train_val_subset.csv