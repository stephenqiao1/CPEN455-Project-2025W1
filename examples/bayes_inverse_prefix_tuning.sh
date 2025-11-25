uv run -m examples.bayes_inverse \
--method prefix_tuning \
--max_seq_len 256 \
--batch_size 8 \
--num_iterations 200 \
--learning_rate 1e-2 \
--prefix_length 20