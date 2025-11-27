#!/usr/bin/env python3
"""
uv run -m autograder.auto_grader
"""

import argparse
from autograder.dataset import CPEN455_2025_W1_Dataset
import pandas as pd

def load_predictions(predictions_results_path: str):
    df = pd.read_csv(predictions_results_path)
    # Remove percentage signs and convert to float
    for col in ["prob_ham", "prob_spam"]:
        df[col] = df[col].astype(str).str.rstrip("%").astype(float)
    # Return list of tuples: (data_index, predicted_label)
    # predicted_label = 1 (spam) if prob_spam > prob_ham, else 0 (ham)
    predictions = {}
    for _, row in df.iterrows():
        data_index = row["data_index"]
        predicted_label = 1 if row["prob_spam"] > row["prob_ham"] else 0
        predictions[int(data_index)] = predicted_label
    return predictions

def calculate_accuracy(predictions, ground_truth_dataset) -> float:
    correct = 0
    total = 0
    
    for data in ground_truth_dataset:
        data_index, _, _, true_label = data
        if true_label == predictions[data_index]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute classification accuracy for classification results.")
    parser.add_argument(
        "--predictions_results_path",
        default="bayes_inverse_probs/test_dataset_probs.csv",
        type=str,
        help="CSV file with columns data_index, prob_ham, prob_spam.",
    )
    parser.add_argument(
        "--ground_truth_path",
        default="autograder/cpen455_released_datasets/test_subset_random_labels.csv",
        type=str,
        help="CSV file with columns Index, Subject, Message, Spam/Ham.",
    )
    args = parser.parse_args()

    gt_dataset = CPEN455_2025_W1_Dataset(csv_path=args.ground_truth_path)
    predictions = load_predictions(args.predictions_results_path)
    
    accuracy = calculate_accuracy(predictions, gt_dataset)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    

if __name__ == "__main__":
    main()
