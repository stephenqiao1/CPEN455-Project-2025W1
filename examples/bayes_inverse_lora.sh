uv run -m examples.bayes_inverse \
--method lora \
--max_seq_len 512 \
--batch_size 8 \
--num_iterations 400 \
--learning_rate 5e-5 \
--lora_rank 32 \
--lora_alpha 64