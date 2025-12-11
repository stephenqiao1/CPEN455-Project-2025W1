#!/usr/bin/env python3

"""
Minimal bayes inverse example for SmolLM2-135M-Instruct.

Filepath: ./examples/bayes_inverse_example.py
Project: CPEN455-Project-2025W1
Description: integrates three different ways to perform bayes inverse classification with LLMs for spam detection.

Usage:
    uv run -m examples.bayes_inverse
"""

import os
import pdb
import json
import wandb
from dotenv import load_dotenv
from einops import rearrange
from tqdm import tqdm
import argparse
import random
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP, prepare_subset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt
from utils.logger import avg_logger, avg_acc_logger
    
def get_seq_log_prob(prompts, tokenizer, model, device):
    encoded_batch = tokenizer.encode(
        prompts, return_tensors="pt", return_attention_mask=True
    )
    
    input_ids = encoded_batch["input_ids"].to(device)
    attention_mask = encoded_batch["attention_mask"].to(device)

    log_prob, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    shifted_log_prob = log_prob[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:]

    gathered_log_prob = shifted_log_prob.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    gathered_log_prob = gathered_log_prob * shifted_attention_mask
    
    return gathered_log_prob.sum(dim=-1)


METHOD_SET = ["zero_shot", "naive_prompting", "full_finetune"]

def is_required_training(method: str) -> bool:
    assert method in METHOD_SET, f"Method {method} not recognized. Choose from {METHOD_SET}."
    return method in METHOD_SET[2:]

def bayes_inverse_llm_classifier(args, model, batch, tokenizer, device):

    _, subjects, messages, labels = batch

    prompts_ham = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[0], max_seq_length=args.max_seq_len, user_prompt=args.user_prompt) for subj, msg in zip(subjects, messages)]
    prompts_spam = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[1], max_seq_length=args.max_seq_len, user_prompt=args.user_prompt) for subj, msg in zip(subjects, messages)]

    # The first half are ham, the second half are spam
    prompts = prompts_ham + prompts_spam
    with torch.no_grad():
        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)

        '''
        Rearrange to (batch_size, 2), in this way, the second dimension 0 is ham, 1 is spam.
        '''
        seq_log_prob = rearrange(seq_log_prob, '(c b) -> b c', c=2)
        
        '''
        Apply softmax over ham/spam dimension to get probabilities.
        The shape of probs will be (2, batch_size), where probs[0, :] is ham probability and probs[1, :] is spam probability.
        probs[:, i] gives the category distribution used to classify spam and ham for the i-th email in the batch.
        '''
        probs = F.softmax(seq_log_prob, dim=-1)

        labels_pred = torch.argmax(probs, dim=-1)
        
        if  -1 in labels:
            is_correct = None
        else:
            is_correct = labels_pred.cpu() == labels

        return is_correct, (probs.detach().cpu(), labels_pred.detach().cpu())

def train_or_test(args, model, tokenizer, batch, optimizer=None, is_training=True):
    """
    Training or testing step with gradient clipping.
    
    Args:
        args: Training arguments
        model: The model to train/test
        tokenizer: Tokenizer
        batch: Input batch
        optimizer: Optimizer (required for training)
        is_training: Whether this is a training step
    """
    if is_training:
        model.train()
    else:
        model.eval()

    # Get device from model
    device_obj = next(model.parameters()).device

    _, subjects, messages, label_indexs = batch
    
    if -1 in label_indexs:
        bpd = None
    else:
        labels_text = [ENRON_LABEL_INDEX_MAP.inv[int(label_index)] for label_index in label_indexs]

        prompts = [get_prompt(subject=subj, message=msg, label=label, max_seq_length=args.max_seq_len) for subj, msg, label in zip(subjects, messages, labels_text)]

        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device=device_obj)
        num_characters = torch.tensor([len(prompt) for prompt in prompts], device=device_obj).sum()
        bpd = -seq_log_prob.sum()/num_characters

        if is_training:
            assert optimizer is not None, "Optimizer must be provided during training."
            optimizer.zero_grad()
            bpd.backward()
            optimizer.step()

    is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device_obj)

    return bpd, is_correct, (probs, labels_pred)

def save_probs(args, model, tokenizer, dataloader, device, name = "test"):
    save_path = os.path.join(os.getcwd(), f"{args.prob_output_folder}/{name}_dataset_probs.csv")
    
    if os.path.exists(save_path):
        os.remove(save_path)
        
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="saving probabilities"):
            
            _, (probs, _) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device = device)
            
            data_index, _, _, _ = batch
            indices = torch.as_tensor(data_index).view(-1).tolist()
            
            rows = zip(indices, probs[:, 0].tolist(), probs[:, 1].tolist())
            
            file_exists = os.path.exists(save_path)
            with open(save_path, "a", newline="") as handle:
                if not file_exists:
                    handle.write("data_index,prob_ham,prob_spam\n")
                handle.writelines(f"{idx},{ham},{spam}\n" for idx, ham, spam in rows)

if __name__ == "__main__":
    # random seed for reproducibility
    torch.manual_seed(25)
    random.seed(25)       # For experiments involving random sampling
    np.random.seed(25)   

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="zero-shot", choices=METHOD_SET)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dataset_path", type=str, default="autograder/cpen455_released_datasets/train_val_subset.csv")
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    
    # Training hyperparameters
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--val_frequency", type=int, default=10,
                        help="Frequency of validation checks during training (default: 10, i.e., every 10 iterations)")
    
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of validation checks to wait before early stopping (None to disable)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum change in validation accuracy to qualify as an improvement")
    
    # Weight decay for regularization
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (L2 regularization) coefficient for optimizer (default: 0.0)")
    
    parser.add_argument("--checkpoint_dir", type=str, default="examples/ckpts",
                        help="Directory to save model checkpoints")
    
    # Dataset usage
    parser.add_argument("--use_full_dataset", action="store_true",
                        help="Use entire dataset for training (no train/val split). Disables validation and early stopping.")
    args = parser.parse_args()

    load_dotenv()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    run = None
    if not is_required_training(args.method):
        run = wandb.init(
            project=os.getenv("PROJECT_NAME"), 
            name=f"bayes-inverse-{args.method}_msl{args.max_seq_len}",
        )
    else:
        run = wandb.init(
            project=os.getenv("PROJECT_NAME"), 
            name=f"bayes-inverse-{args.method}_msl{args.max_seq_len}_ni{args.num_iterations}_bs{args.batch_size}",
        )
        
    wandb.config.update(args)

    # Set device to GPU if available, to MPS if on Mac with M-series chip, else CPU
    device = set_device()

    # Load tokenizer and config
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Load model
    model = LlamaModel(config)
    load_model_weights(model, checkpoint, cache_dir=model_cache_dir, device=device)
    model = model.to(device)

    # Set up datasets and dataloaders
    train_n_val_dataset = CPEN455_2025_W1_Dataset(csv_path=args.dataset_path)
    
    if args.use_full_dataset:
        # Use entire dataset for training (no validation split)
        print(f"Using full dataset for training ({len(train_n_val_dataset)} samples)")
        training_dataset = train_n_val_dataset
        val_dataset = None
    else:
        # Split into training (80%) and validation (20%)
        training_dataset, val_dataset = prepare_subset(train_n_val_dataset, int(0.8 * len(train_n_val_dataset)), ratio_spam=0.5, return_remaining=True)
        print(f"Training dataset: {len(training_dataset)} samples, Validation dataset: {len(val_dataset)} samples")
    
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)

    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
        )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        ) if val_dataset is not None else None
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if args.weight_decay > 0:
        print(f"Weight decay enabled: {args.weight_decay}")
    else:
        print("Weight decay disabled (weight_decay=0.0)")
    
    if os.path.exists(args.prob_output_folder) == False:
        os.makedirs(args.prob_output_folder)
    
    if os.path.exists(args.checkpoint_dir) == False:
        os.makedirs(args.checkpoint_dir)
    
    # Early stopping setup
    # Disable early stopping if using full dataset (no validation set available)
    early_stopping_enabled = (not args.use_full_dataset and
                              args.early_stopping_patience is not None and 
                              args.early_stopping_patience > 0 and 
                              is_required_training(args.method))
    
    if args.use_full_dataset:
        print("Validation and early stopping disabled (using full dataset)")
    
    best_val_accuracy = float('-inf')
    best_val_bpd = float('inf')
    patience_counter = 0
    best_model_state = None
    best_checkpoint_path = None
    best_iteration = None  # Track which iteration had the best model
    early_stopping_triggered = False  # Track if early stopping was triggered
    early_stopping_iteration = None  # Track when early stopping was triggered
    iterations_completed = 0  # Track total iterations completed
    
    if early_stopping_enabled:
        print(f"Early stopping enabled: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
        print(f"Monitoring: validation accuracy (maximize) and validation BPD (minimize)")

    for iteration in tqdm(range(args.num_iterations), desc="Training"):
                    
        if (iteration + 1) % args.val_frequency == 0 and not args.use_full_dataset:
            val_acc_logger = avg_acc_logger()
            val_bpd_logger = avg_logger()
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Evaluating on validation set during training"):
                    
                    bpd, is_correct, (probs, labels_pred) = train_or_test(
                        args = args, 
                        model = model, 
                        tokenizer = tokenizer, 
                        batch = batch, 
                        is_training=False)
                    
                    val_acc_logger.update(is_correct)
                    val_bpd_logger.update(bpd.item())

            val_accuracy = val_acc_logger.compute_accuracy()
            val_bpd = val_bpd_logger.compute_average()
            
            wandb.log({
                "val_avg_bpd": val_bpd,
                "val_avg_accuracy": val_accuracy,
                "training_iteration": iteration,
            })
            
            print(f"Validation accuracy: {val_accuracy:.4f}, Validation BPD: {val_bpd:.4f}")
            
            # Early stopping logic
            if early_stopping_enabled:
                # Check if validation accuracy improved (higher is better)
                acc_improvement = val_accuracy - best_val_accuracy
                # Check if validation BPD improved (lower is better)
                bpd_improvement = best_val_bpd - val_bpd
                
                # Model improved if either accuracy increased OR BPD decreased
                accuracy_improved = acc_improvement > args.early_stopping_min_delta
                bpd_improved = bpd_improvement > args.early_stopping_min_delta
                
                if accuracy_improved or bpd_improved:
                    # New best model found
                    if accuracy_improved:
                        best_val_accuracy = val_accuracy
                    if bpd_improved:
                        best_val_bpd = val_bpd
                    
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_iteration = iteration + 1  # Track the iteration where best model was found
                    
                    # Save best model checkpoint
                    if best_checkpoint_path is None:
                        best_checkpoint_path = os.path.join(args.checkpoint_dir, f"model_{args.method}_best.pt")
                    torch.save(best_model_state, best_checkpoint_path)
                    
                    improvements = []
                    if accuracy_improved:
                        improvements.append(f"accuracy: {val_accuracy:.4f} (Δ+{acc_improvement:.4f})")
                    if bpd_improved:
                        improvements.append(f"BPD: {val_bpd:.4f} (Δ-{bpd_improvement:.4f})")
                    print(f"New best model! {' | '.join(improvements)}")
                    print(f"Saved best model checkpoint to {best_checkpoint_path}")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} validation check(s) (best acc: {best_val_accuracy:.4f}, best BPD: {best_val_bpd:.4f})")
                    
                    # Check if we should stop early
                    if patience_counter >= args.early_stopping_patience:
                        early_stopping_triggered = True
                        early_stopping_iteration = iteration + 1
                        best_iteration_at_stop = best_iteration if best_iteration is not None else (iteration + 1 - patience_counter * args.val_frequency)
                        
                        print(f"\nEarly stopping triggered after {early_stopping_iteration} iterations!")
                        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
                        print(f"Best validation BPD: {best_val_bpd:.4f}")
                        print(f"Best model was at iteration {best_iteration_at_stop}")
                        print(f"Restoring best model weights...")
                        
                        # Log to wandb
                        wandb.log({
                            "early_stopping_triggered": True,
                            "early_stopping_iteration": early_stopping_iteration,
                            "best_iteration": best_iteration_at_stop,
                            "best_val_accuracy": best_val_accuracy,
                            "best_val_bpd": best_val_bpd,
                        })
                        
                        # Restore best model
                        model.load_state_dict(best_model_state)
                        model = model.to(device)
                        break
                    
        if not is_required_training(args.method):
            break
                    
        batch = next(iter(training_dataloader))
        
        bpd, is_correct, _ = train_or_test(
            args = args, 
            model = model, 
            tokenizer = tokenizer, 
            batch = batch, 
            optimizer = optimizer,
            is_training = True)
        
        wandb.log({
            "training_batch_bpd": bpd.item(),
            "training_batch_acc": is_correct.float().mean().item(),
            "training_iteration": iteration,
            })
        
        iterations_completed = iteration + 1

    # Save final checkpoint when using full dataset
    if args.use_full_dataset:
        final_checkpoint_path = os.path.join(args.checkpoint_dir, f"model_{args.method}_full_dataset_final.pt")
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"\nTraining completed. Final model saved to {final_checkpoint_path}")
    
    # Restore best model if early stopping was enabled and we didn't already restore it
    if early_stopping_enabled and best_model_state is not None:
        # Only restore if we didn't break early (i.e., we completed all iterations)
        if patience_counter < args.early_stopping_patience:
            print(f"\nTraining completed. Restoring best model...")
            print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"  Best validation BPD: {best_val_bpd:.4f}")
            model.load_state_dict(best_model_state)
            model = model.to(device)
        else:
            # We broke early, model was already restored
            print(f"\nTraining stopped early. Best model restored.")
            print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"  Best validation BPD: {best_val_bpd:.4f}")
    
    # Save early stopping information even if training completed without triggering
    if early_stopping_enabled:
        early_stopping_info = {
            "early_stopping_enabled": True,
            "early_stopping_triggered": early_stopping_triggered,
            "early_stopping_iteration": early_stopping_iteration,
            "best_iteration": best_iteration,
            "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy > float('-inf') else None,
            "best_val_bpd": float(best_val_bpd) if best_val_bpd < float('inf') else None,
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "total_iterations": args.num_iterations,
            "iterations_completed": iterations_completed,
            "best_checkpoint_path": best_checkpoint_path,
            "timestamp": datetime.now().isoformat(),
        }
        
        early_stopping_file = os.path.join(args.checkpoint_dir, f"early_stopping_{args.method}.json")
        with open(early_stopping_file, 'w') as f:
            json.dump(early_stopping_info, f, indent=2)
        print(f"Early stopping information saved to {early_stopping_file}")
        
        # Log to wandb
        wandb.log({
            "early_stopping_info": early_stopping_info,
            "best_iteration": best_iteration,
            "early_stopping_triggered": early_stopping_triggered,
        })

    # Save final model checkpoint
    if is_required_training(args.method):
        final_checkpoint_path = os.path.join(args.checkpoint_dir, f"model_{args.method}_final.pt")
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Saving final model checkpoint to {final_checkpoint_path}")
        
        # Print summary if early stopping was used
        if early_stopping_enabled and best_val_accuracy > float('-inf'):
            print(f"Best validation accuracy during training: {best_val_accuracy:.4f}")
            print(f"Best validation BPD during training: {best_val_bpd:.4f}")
    
    # After training, save probabilities on test set
    train_n_val_dataloader = DataLoader(
        train_n_val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    save_probs(args, model, tokenizer, train_n_val_dataloader, device=device, name = "train_n_val")
    save_probs(args, model, tokenizer, test_dataloader, device=device, name = "test")