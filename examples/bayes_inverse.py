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
import wandb
from dotenv import load_dotenv
from einops import rearrange
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP, prepare_subset
from model import LlamaModel
from model.prefix_llama import PrefixLlamaModel
from model.lora import apply_lora_to_model
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


METHOD_SET = ["zero_shot", "naive_prompting", "full_finetune", "prefix_tuning", "lora"]

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
    if is_training:
        model.train()
    else:
        model.eval()

    _, subjects, messages, label_indexs = batch
    
    if -1 in label_indexs:
        bpd = None
    else:
        labels_text = [ENRON_LABEL_INDEX_MAP.inv[int(label_index)] for label_index in label_indexs]

        prompts = [get_prompt(subject=subj, message=msg, label=label, max_seq_length=args.max_seq_len) for subj, msg, label in zip(subjects, messages, labels_text)]

        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device=device)
        
        num_characters = torch.tensor([len(prompt) for prompt in prompts], device=device).sum()
        bpd = -seq_log_prob.sum()/num_characters

        if is_training:
            assert optimizer is not None, "Optimizer must be provided during training."
            optimizer.zero_grad()
            bpd.backward()
            optimizer.step()

    is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device)

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


def save_checkpoint(model, checkpoint_path, method):
    """
    Save model checkpoint after training.
    
    Args:
        model: The trained model (could be LlamaModel, PrefixLlamaModel, or LoRA-wrapped model)
        checkpoint_path: Path to save the checkpoint
        method: Training method used (full_finetune, prefix_tuning, lora)
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    if method == "prefix_tuning":
        # For prefix tuning, save the base model's state dict
        # The prefix parameters are part of the PrefixLlamaModel
        state_dict = model.model.state_dict()
        print(f"Saving base model state dict for prefix tuning to: {checkpoint_path}")
    elif method == "lora":
        # For LoRA, save the entire model state dict (includes LoRA parameters)
        state_dict = model.state_dict()
        print(f"Saving LoRA model state dict to: {checkpoint_path}")
    else:
        # For full finetune, save the entire model state dict
        state_dict = model.state_dict()
        print(f"Saving full model state dict to: {checkpoint_path}")
    
    torch.save(state_dict, checkpoint_path)
    print(f"Checkpoint saved successfully to: {checkpoint_path}")


class EarlyStopping:
    """
    Early stopping to stop training when validation performance doesn't improve.
    
    Args:
        patience: Number of evaluations to wait before stopping after no improvement
        min_delta: Minimum change in monitored metric to qualify as an improvement
        mode: 'max' for metrics where higher is better (e.g., accuracy),
              'min' for metrics where lower is better (e.g., loss)
        verbose: Whether to print messages
    """
    def __init__(self, patience=5, min_delta=0.0, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_iteration = 0
        
    def __call__(self, score, iteration):
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            iteration: Current training iteration
            
        Returns:
            bool: True if this is a new best score, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_iteration = iteration
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta
            
        if improved:
            if self.verbose:
                print(f"Validation metric improved from {self.best_score:.4f} to {score:.4f}")
            self.best_score = score
            self.best_iteration = iteration
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} evaluations. "
                      f"Best: {self.best_score:.4f} at iteration {self.best_iteration}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best score: {self.best_score:.4f} "
                          f"at iteration {self.best_iteration}")
            return False


if __name__ == "__main__":
    # random seed for reproducibility
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="zero-shot", choices=METHOD_SET)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dataset_path", type=str, default="autograder/cpen455_released_datasets/train_val_subset.csv")
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    # Training hyperparameters
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--prefix_length", type=int, default=20)
    
    # Checkpoint saving arguments
    parser.add_argument("--save_checkpoint", action="store_true", help="Save model checkpoint after training")
    parser.add_argument("--checkpoint_path", type=str, default="examples/ckpts/model_finetuned.pt", help="Path to save the checkpoint")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Number of evaluations to wait before early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum improvement to reset patience counter")
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate every N iterations")
    
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

    # Wrap the model for prefix tuning if needed
    if args.method == "prefix_tuning":
        prefix_length = args.prefix_length
        model = PrefixLlamaModel(base_model=model, prefix_length=prefix_length)
    elif args.method == "lora":
        apply_lora_to_model(model, r=args.lora_rank, lora_alpha=args.lora_alpha)

    model = model.to(device)

    # Set up datasets and dataloaders
    train_n_val_dataset = CPEN455_2025_W1_Dataset(csv_path=args.dataset_path)
    training_dataset, val_dataset = prepare_subset(train_n_val_dataset, int(0.8 * len(train_n_val_dataset)), ratio_spam=0.5, return_remaining=True)
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
        )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    
    # If using Prefix tuning
    if args.method == "prefix_tuning":
        params_to_optimize = model.trainable_parameters()
    else:
        params_to_optimize = model.parameters()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    if os.path.exists(args.prob_output_folder) == False:
        os.makedirs(args.prob_output_folder)

    # Initialize early stopping if enabled
    early_stopper = None
    if args.early_stopping and is_required_training(args.method):
        early_stopper = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode='max',  # We're maximizing accuracy
            verbose=True
        )
        print(f"Early stopping enabled with patience={args.patience}, min_delta={args.min_delta}")

    # Track best validation accuracy for saving best checkpoint
    best_val_acc = 0.0
    final_iteration = 0

    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        final_iteration = iteration
        
        # Evaluation phase
        if (iteration + 1) % args.eval_interval == 0:
            val_acc_logger = avg_acc_logger()
            val_bpd_logger = avg_logger()
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Evaluating on validation set during training", leave=False):
                    
                    bpd, is_correct, (probs, labels_pred) = train_or_test(
                        args = args, 
                        model = model, 
                        tokenizer = tokenizer, 
                        batch = batch, 
                        is_training=False)
                    
                    val_acc_logger.update(is_correct)
                    val_bpd_logger.update(bpd.item())

            current_val_acc = val_acc_logger.compute_accuracy()
            current_val_bpd = val_bpd_logger.compute_average()
            
            wandb.log({
                "val_avg_bpd": current_val_bpd,
                "val_avg_accuracy": current_val_acc,
                "training_iteration": iteration,
            })
            
            print(f"\nIteration {iteration + 1}: Val Accuracy = {current_val_acc:.4f}, Val BPD = {current_val_bpd:.4f}")
            
            # Check early stopping
            if early_stopper is not None:
                is_best = early_stopper(current_val_acc, iteration)
                
                # Save best checkpoint
                if is_best and args.save_checkpoint:
                    best_val_acc = current_val_acc
                    best_checkpoint_path = args.checkpoint_path.replace(".pt", "_best_val.pt")
                    save_checkpoint(model, best_checkpoint_path, args.method)
                
                # Check if we should stop
                if early_stopper.early_stop:
                    print(f"\nEarly stopping at iteration {iteration + 1}!")
                    print(f"Best validation accuracy: {early_stopper.best_score:.4f} at iteration {early_stopper.best_iteration + 1}")
                    break
            else:
                # Without early stopping, still save best checkpoint
                if args.save_checkpoint and current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_checkpoint_path = args.checkpoint_path.replace(".pt", "_best_val.pt")
                    save_checkpoint(model, best_checkpoint_path, args.method)
                    print(f"New best validation accuracy: {best_val_acc:.4f}")
                    
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

    # Print training summary
    if is_required_training(args.method):
        print(f"\n{'='*50}")
        print("Training Summary:")
        print(f"{'='*50}")
        print(f"Total iterations: {final_iteration + 1}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        if early_stopper is not None and early_stopper.early_stop:
            print(f"Training stopped early at iteration {final_iteration + 1}")
            print(f"Best model was at iteration {early_stopper.best_iteration + 1}")
        print(f"{'='*50}\n")

    # Save final checkpoint after training if requested
    if args.save_checkpoint and is_required_training(args.method):
        save_checkpoint(model, args.checkpoint_path, args.method)
        print(f"Final checkpoint saved to: {args.checkpoint_path}")

    # After training, save probabilities on test set
    train_n_val_dataloader = DataLoader(
        train_n_val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    save_probs(args, model, tokenizer, train_n_val_dataloader, device=device, name = "train_n_val")
    save_probs(args, model, tokenizer, test_dataloader, device=device, name = "test")