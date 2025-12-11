# This config spends about 3 minutes to run on a Macmini with M4 Chip, 16GB RAM
uv run -m examples.bayes_inverse \
--method full_finetune \
--use_full_dataset \
--max_seq_len 256 \
--batch_size 4 \
--num_iterations 200 \
--learning_rate 5e-6 \
--weight_decay 0.001 \
--early_stopping_patience 5 \
--early_stopping_min_delta 0.001 \
--val_frequency 1 \
--checkpoint_dir examples/ckpts
# best right now is 5e-6 lr and num_iteration 300 and batch_size 4 with early stopping,
# weight decay of 0.001