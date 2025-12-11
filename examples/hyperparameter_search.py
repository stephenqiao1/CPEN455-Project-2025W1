#!/usr/bin/env python3

"""
Hyperparameter search script for batch size and weight decay.

This script tests different combinations of batch_size and weight_decay
and tracks validation accuracy to find the best configuration.
"""

import os
import subprocess
import json
import time
import shutil
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from autograder.dataset import CPEN455_2025_W1_Dataset
from model import LlamaModel
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from examples.bayes_inverse import save_probs
from torch.utils.data import DataLoader

# Hyperparameter configurations to test
BATCH_SIZES = [4, 16]  # Test both batch sizes
WEIGHT_DECAYS = [0.0, 0.0005, 0.001, 0.0015, 0.002]  # Range around best value
LEARNING_RATES = [5e-6, 1e-5, 2e-5]  # Test different learning rates

# Fixed hyperparameters
NUM_ITERATIONS = 300
MAX_SEQ_LEN = 256
METHOD = "full_finetune"
CHECKPOINT_DIR = "examples/ckpts"

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 5  # Number of validation checks (each check is every 10 iterations, so 5 = 50 iterations)
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement in validation accuracy

# Results storage
results = []
results_file = "examples/hyperparameter_search_results.json"

# Test dataset path for evaluation
TEST_DATASET_PATH = "autograder/cpen455_released_datasets/test_subset.csv"
PROB_OUTPUT_FOLDER = "bayes_inverse_probs"

def evaluate_checkpoint(checkpoint_path, batch_size, max_seq_len):
    """
    Evaluate a checkpoint by loading it and computing average probabilities on test set.
    Returns average prob_ham, average prob_spam, and distance from 0.5/0.5.
    Closer to 0.5/0.5 is better.
    """
    try:
        load_dotenv()
        checkpoint = os.getenv("MODEL_CHECKPOINT")
        model_cache_dir = os.getenv("MODEL_CACHE_DIR")
        
        device = set_device()
        
        # Load tokenizer and config
        tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
        base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
        config = Config._find_config_files(base_path)
        
        # Load model
        model = LlamaModel(config)
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Load test dataset
        test_dataset = CPEN455_2025_W1_Dataset(csv_path=TEST_DATASET_PATH)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Create a temporary args object for save_probs
        class Args:
            def __init__(self):
                self.max_seq_len = max_seq_len
                self.prob_output_folder = PROB_OUTPUT_FOLDER
                self.user_prompt = ""
        
        args = Args()
        
        # Ensure output folder exists
        os.makedirs(PROB_OUTPUT_FOLDER, exist_ok=True)
        
        # Save probabilities to a unique file for this evaluation
        # Extract identifier from checkpoint path
        checkpoint_basename = os.path.basename(checkpoint_path).replace('.pt', '')
        eval_output_name = f"eval_{checkpoint_basename}"
        save_probs(args, model, tokenizer, test_dataloader, device=device, name=eval_output_name)
        
        # Read the probabilities file
        prob_file = os.path.join(PROB_OUTPUT_FOLDER, f"{eval_output_name}_dataset_probs.csv")
        if not os.path.exists(prob_file):
            print(f"Warning: Probability file not found: {prob_file}")
            return None, None, float('inf')
        
        # Calculate average probabilities
        df = pd.read_csv(prob_file)
        avg_prob_ham = df['prob_ham'].mean()
        avg_prob_spam = df['prob_spam'].mean()
        
        # Calculate distance from ideal 0.5/0.5 (closer is better)
        # Use L2 distance: sqrt((prob_ham - 0.5)^2 + (prob_spam - 0.5)^2)
        distance_from_ideal = ((avg_prob_ham - 0.5) ** 2 + (avg_prob_spam - 0.5) ** 2) ** 0.5
        
        print(f"  Average prob_ham: {avg_prob_ham:.4f}, Average prob_spam: {avg_prob_spam:.4f}")
        print(f"  Distance from 0.5/0.5: {distance_from_ideal:.4f} (lower is better)")
        
        return avg_prob_ham, avg_prob_spam, distance_from_ideal
        
    except Exception as e:
        print(f"Error evaluating checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None, float('inf')

def run_experiment(batch_size, weight_decay, learning_rate):
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Testing: batch_size={batch_size}, weight_decay={weight_decay}, learning_rate={learning_rate}")
    print(f"{'='*60}")
    
    # Create unique checkpoint filename based on hyperparameters
    # Format: model_full_finetune_bs{bs}_wd{wd}_lr{lr}.pt
    # Format weight decay: 0.01 -> "0.01", 0.001 -> "0.001", 0.0005 -> "0.0005", 0.0 -> "0.0"
    if weight_decay == 0.0:
        wd_str = "0.0"
    elif weight_decay >= 1:
        wd_str = f"{weight_decay:.0f}"
    else:
        # Format as decimal with enough precision to avoid collisions
        # Use up to 4 decimal places, then strip trailing zeros
        wd_str = f"{weight_decay:.4f}".rstrip('0').rstrip('.')
    
    # Format learning rate: 5e-6 -> "5e-6", 1e-5 -> "1e-5", 2e-5 -> "2e-5"
    if learning_rate >= 1:
        lr_str = f"{learning_rate:.0f}"
    else:
        # Format as scientific notation: 5e-6, 1e-5, etc.
        lr_str = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    checkpoint_filename = f"model_{METHOD}_bs{batch_size}_wd{wd_str}_lr{lr_str}.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    
    cmd = [
        "uv", "run", "-m", "examples.bayes_inverse",
        "--method", METHOD,
        "--max_seq_len", str(MAX_SEQ_LEN),
        "--batch_size", str(batch_size),
        "--num_iterations", str(NUM_ITERATIONS),
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
        "--early_stopping_min_delta", str(EARLY_STOPPING_MIN_DELTA),
        "--checkpoint_dir", CHECKPOINT_DIR
    ]
    
    start_time = time.time()
    try:
        # Use Popen to stream output in real-time while capturing it
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output while streaming it in real-time
        stdout_lines = []
        val_accuracy = None
        val_accuracies = []
        
        # Read and print output line by line in real-time
        for line in process.stdout:
            # Print immediately to show progress
            print(line, end='')
            stdout_lines.append(line)
            
            # Parse validation accuracy from output in real-time
            # Check for best validation accuracy (from early stopping summary)
            if "Best validation accuracy during training:" in line:
                try:
                    val_accuracy = float(line.split("Best validation accuracy during training:")[1].strip())
                except (ValueError, IndexError):
                    pass
            # Also collect all validation accuracies
            elif "Validation accuracy:" in line and "New best" not in line:
                try:
                    acc = float(line.split("Validation accuracy:")[1].strip())
                    val_accuracies.append(acc)
                except (ValueError, IndexError):
                    pass
        
        # Wait for process to complete
        return_code = process.wait()
        elapsed_time = time.time() - start_time
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        # Combine all output for final parsing (in case we missed something)
        stdout_text = ''.join(stdout_lines)
        
        # Final parse if we didn't get accuracy yet
        if val_accuracy is None:
            for line in stdout_text.split('\n'):
                # Check for best validation accuracy (from early stopping summary)
                if "Best validation accuracy during training:" in line:
                    try:
                        val_accuracy = float(line.split("Best validation accuracy during training:")[1].strip())
                        break  # Use this as it's the best one
                    except (ValueError, IndexError):
                        pass
                # Also collect all validation accuracies
                elif "Validation accuracy:" in line and "New best" not in line:
                    try:
                        acc = float(line.split("Validation accuracy:")[1].strip())
                        val_accuracies.append(acc)
                    except (ValueError, IndexError):
                        pass
        
        # Use best accuracy if found, otherwise use the last validation accuracy
        if val_accuracy is None:
            if val_accuracies:
                val_accuracy = val_accuracies[-1]
            else:
                print("Warning: Could not parse validation accuracy from output")
                val_accuracy = 0.0
        
        # Rename the checkpoint to include hyperparameters
        # Prefer best checkpoint (from early stopping) over final checkpoint
        best_checkpoint = os.path.join(CHECKPOINT_DIR, f"model_{METHOD}_best.pt")
        final_checkpoint = os.path.join(CHECKPOINT_DIR, f"model_{METHOD}_final.pt")
        
        source_checkpoint = None
        if os.path.exists(best_checkpoint):
            source_checkpoint = best_checkpoint
            print(f"Using best checkpoint from early stopping")
        elif os.path.exists(final_checkpoint):
            source_checkpoint = final_checkpoint
            print(f"Using final checkpoint")
        
        if source_checkpoint:
            if os.path.exists(checkpoint_path):
                # Remove old checkpoint if it exists
                os.remove(checkpoint_path)
            shutil.copy2(source_checkpoint, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
        else:
            print(f"Warning: Expected checkpoint not found at {best_checkpoint} or {final_checkpoint}")
            checkpoint_path = None
        
        # Evaluate the checkpoint by running inference
        avg_prob_ham = None
        avg_prob_spam = None
        distance_from_ideal = float('inf')
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nEvaluating checkpoint: {checkpoint_path}")
            avg_prob_ham, avg_prob_spam, distance_from_ideal = evaluate_checkpoint(
                checkpoint_path, batch_size, MAX_SEQ_LEN
            )
        
        return {
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "validation_accuracy": val_accuracy,
            "avg_prob_ham": avg_prob_ham,
            "avg_prob_spam": avg_prob_spam,
            "distance_from_ideal": distance_from_ideal,
            "elapsed_time": elapsed_time,
            "checkpoint_path": checkpoint_path,
            "status": "success",
            "stdout": stdout_text[-500:],  # Last 500 chars for debugging
        }
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\nError running experiment: {e}")
        return {
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "validation_accuracy": 0.0,
            "elapsed_time": elapsed_time,
            "checkpoint_path": None,
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nUnexpected error running experiment: {e}")
        return {
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "validation_accuracy": 0.0,
            "elapsed_time": elapsed_time,
            "checkpoint_path": None,
            "status": "error",
            "error": str(e),
        }

def save_results():
    """Save results to JSON file."""
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_iterations": NUM_ITERATIONS,
                "max_seq_len": MAX_SEQ_LEN,
                "method": METHOD,
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

def print_summary():
    """Print summary of all results."""
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("No results to display.")
        return
    
    # Sort by distance from ideal (lower is better), then by validation accuracy
    sorted_results = sorted(results, key=lambda x: (
        x.get("distance_from_ideal", float('inf')),
        -x.get("validation_accuracy", 0.0)  # Negative for reverse sort
    ))
    
    print(f"\nTop 5 Configurations (sorted by distance from 0.5/0.5, lower is better):")
    print(f"{'Rank':<6} {'Batch':<8} {'Weight Decay':<15} {'Learning Rate':<15} {'Dist 0.5/0.5':<15} {'Avg Prob Ham':<15} {'Avg Prob Spam':<15} {'Val Acc':<10}")
    print("-" * 110)
    
    for i, result in enumerate(sorted_results[:5], 1):
        dist = result.get('distance_from_ideal', float('inf'))
        prob_ham = result.get('avg_prob_ham', 0.0)
        prob_spam = result.get('avg_prob_spam', 0.0)
        val_acc = result.get('validation_accuracy', 0.0)
        lr = result.get('learning_rate', 0.0)
        
        dist_str = f"{dist:.4f}" if dist < float('inf') else "N/A"
        prob_ham_str = f"{prob_ham:.4f}" if prob_ham is not None else "N/A"
        prob_spam_str = f"{prob_spam:.4f}" if prob_spam is not None else "N/A"
        lr_str = f"{lr:.2e}" if lr > 0 else "N/A"
        
        print(f"{i:<6} {result['batch_size']:<8} {result.get('weight_decay', 0.0):<15.4f} "
              f"{lr_str:<15} {dist_str:<15} {prob_ham_str:<15} {prob_spam_str:<15} {val_acc:<10.4f}")
    
    # Best configuration
    best = sorted_results[0]
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION (closest to 0.5/0.5):")
    print(f"  Batch Size: {best['batch_size']}")
    print(f"  Weight Decay: {best.get('weight_decay', 0.0)}")
    print(f"  Learning Rate: {best.get('learning_rate', 0.0):.2e}")
    print(f"  Distance from 0.5/0.5: {best.get('distance_from_ideal', float('inf')):.4f}")
    print(f"  Average prob_ham: {best.get('avg_prob_ham', 0.0):.4f}")
    print(f"  Average prob_spam: {best.get('avg_prob_spam', 0.0):.4f}")
    print(f"  Validation Accuracy: {best.get('validation_accuracy', 0.0):.4f}")
    print(f"  Training Time: {best.get('elapsed_time', 0):.1f}s")
    if best.get('checkpoint_path'):
        print(f"  Checkpoint: {best['checkpoint_path']}")
    print(f"{'='*60}")
    
    # List all checkpoints
    print(f"\nAll Checkpoints Saved:")
    for result in sorted_results:
        if result.get('checkpoint_path'):
            print(f"  {os.path.basename(result['checkpoint_path'])} - "
                  f"Val Acc: {result.get('validation_accuracy', 0.0):.4f}")

def main():
    """Run hyperparameter search."""
    print("Starting Hyperparameter Search")
    print(f"Testing {len(BATCH_SIZES)} batch sizes × {len(WEIGHT_DECAYS)} weight decay values × {len(LEARNING_RATES)} learning rates = "
          f"{len(BATCH_SIZES) * len(WEIGHT_DECAYS) * len(LEARNING_RATES)} configurations")
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    total_experiments = len(BATCH_SIZES) * len(WEIGHT_DECAYS) * len(LEARNING_RATES)
    experiment_num = 0
    
    for batch_size in BATCH_SIZES:
        for weight_decay in WEIGHT_DECAYS:
            for learning_rate in LEARNING_RATES:
                experiment_num += 1
                print(f"\n[{experiment_num}/{total_experiments}] ", end="")
                
                result = run_experiment(batch_size, weight_decay, learning_rate)
                results.append(result)
                
                # Save results after each experiment (in case of interruption)
                save_results()
                
                dist = result.get('distance_from_ideal', float('inf'))
                prob_ham = result.get('avg_prob_ham', 0.0)
                prob_spam = result.get('avg_prob_spam', 0.0)
                
                if dist < float('inf'):
                    print(f"Result: Distance from 0.5/0.5 = {dist:.4f}, "
                          f"Avg prob_ham = {prob_ham:.4f}, Avg prob_spam = {prob_spam:.4f}, "
                          f"Val Acc = {result.get('validation_accuracy', 0.0):.4f}, "
                          f"Time = {result.get('elapsed_time', 0):.1f}s")
                else:
                    print(f"Result: Val Accuracy = {result.get('validation_accuracy', 0.0):.4f}, "
                          f"Time = {result.get('elapsed_time', 0):.1f}s")
    
    print_summary()
    save_results()

if __name__ == "__main__":
    main()

