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
from datetime import datetime
from pathlib import Path

# Hyperparameter configurations to test
# Re-running best configurations with fixed checkpoint naming
BATCH_SIZES = [4]  # Focus on best batch size
WEIGHT_DECAYS = [0.0005, 0.0007, 0.0015]  # Top 3 configurations that achieved 100% accuracy

# Fixed hyperparameters
NUM_ITERATIONS = 300
MAX_SEQ_LEN = 256
METHOD = "full_finetune"
CHECKPOINT_DIR = "examples/ckpts"
LEARNING_RATE = 5e-6  # Fixed learning rate (based on previous best results)

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 5  # Number of validation checks (each check is every 10 iterations, so 5 = 50 iterations)
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement in validation accuracy

# Results storage
results = []
results_file = "examples/hyperparameter_search_results.json"

def run_experiment(batch_size, weight_decay):
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Testing: batch_size={batch_size}, weight_decay={weight_decay}")
    print(f"{'='*60}")
    
    # Create unique checkpoint filename based on hyperparameters
    # Format: model_full_finetune_bs{bs}_wd{wd}.pt
    # Format weight decay: 0.01 -> "0.01", 0.001 -> "0.001", 0.0005 -> "0.0005", 0.0 -> "0.0"
    if weight_decay == 0.0:
        wd_str = "0.0"
    elif weight_decay >= 1:
        wd_str = f"{weight_decay:.0f}"
    else:
        # Format as decimal with enough precision to avoid collisions
        # Use up to 4 decimal places, then strip trailing zeros
        wd_str = f"{weight_decay:.4f}".rstrip('0').rstrip('.')
    checkpoint_filename = f"model_{METHOD}_bs{batch_size}_wd{wd_str}.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    
    cmd = [
        "uv", "run", "-m", "examples.bayes_inverse",
        "--method", METHOD,
        "--max_seq_len", str(MAX_SEQ_LEN),
        "--batch_size", str(batch_size),
        "--num_iterations", str(NUM_ITERATIONS),
        "--learning_rate", str(LEARNING_RATE),
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
        
        return {
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "learning_rate": LEARNING_RATE,
            "validation_accuracy": val_accuracy,
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
            "learning_rate": LEARNING_RATE,
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
            "learning_rate": LEARNING_RATE,
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
    
    # Sort by validation accuracy
    sorted_results = sorted(results, key=lambda x: x.get("validation_accuracy", 0.0), reverse=True)
    
    print(f"\nTop 5 Configurations:")
    print(f"{'Rank':<6} {'Batch Size':<12} {'Weight Decay':<15} {'Val Accuracy':<15} {'Time (s)':<10}")
    print("-" * 60)
    
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i:<6} {result['batch_size']:<12} {result.get('weight_decay', 0.0):<15.4f} "
              f"{result.get('validation_accuracy', 0.0):<15.4f} {result.get('elapsed_time', 0):<10.1f}")
    
    # Best configuration
    best = sorted_results[0]
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION:")
    print(f"  Batch Size: {best['batch_size']}")
    print(f"  Weight Decay: {best.get('weight_decay', 0.0)}")
    print(f"  Learning Rate: {best.get('learning_rate', LEARNING_RATE)}")
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
    print(f"Testing {len(BATCH_SIZES)} batch sizes × {len(WEIGHT_DECAYS)} weight decay values = "
          f"{len(BATCH_SIZES) * len(WEIGHT_DECAYS)} configurations")
    print(f"Fixed learning rate: {LEARNING_RATE}")
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    total_experiments = len(BATCH_SIZES) * len(WEIGHT_DECAYS)
    experiment_num = 0
    
    for batch_size in BATCH_SIZES:
        for weight_decay in WEIGHT_DECAYS:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] ", end="")
            
            result = run_experiment(batch_size, weight_decay)
            results.append(result)
            
            # Save results after each experiment (in case of interruption)
            save_results()
            
            print(f"Result: Val Accuracy = {result.get('validation_accuracy', 0.0):.4f}, "
                  f"Time = {result.get('elapsed_time', 0):.1f}s")
    
    print_summary()
    save_results()

if __name__ == "__main__":
    main()

