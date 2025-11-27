# Please save the predicted probabilities on the test dataset to
# 'bayes_inverse_probs/test_dataset_probs.csv' before running the autograder.
uv run -m examples.save_prob_example

# Here we use a synsthetic dataset with random labels, so that you can check if your code runs correctly.
uv run -m autograder.auto_grader \
--predictions_results_path bayes_inverse_probs/test_dataset_probs.csv \
--ground_truth_path autograder/cpen455_released_datasets/test_subset_random_labels.csv

# During grading, we will run the following command to evaluate your submission:
# uv run -m autograder.auto_grader \
# --predictions_results_path bayes_inverse_probs/test_dataset_probs.csv \
# --ground_truth_path autograder/cpen455_released_datasets/test_subset_with_labels.csv