#!/usr/bin/env python3

"""
Weighted Product of Experts (wPoE) implementation for spam detection.
Combines LLaMA (Neural Expert) with N-gram (Statistical Expert) predictions.

Filepath: ./examples/bayes_inverse_wpoe.py
Project: CPEN455-Project-2025W1
Description: Implements wPoE ensemble method based on "Test-Time Steering for Lossless Text Compression via Weighted Product of Experts"

Usage:
    uv run examples/bayes_inverse_wpoe.py --wpoe_alpha 0.7
"""

import os
import sys
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from einops import rearrange
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP
from model import LlamaModel
from model.config import Config
from model.tokenizer import Tokenizer
from model.ngram import NgramExpert
from utils.weight_utils import load_model_weights
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt
from examples.bayes_inverse import get_seq_log_prob


def wpoe_classifier(args, model, batch, tokenizer, device, spam_expert, ham_expert):
    """
    Classifies emails using Weighted Product of Experts (LLM + Ngram).
    """
    _, subjects, messages, labels = batch

    # --- 1. Get LLM Scores (Neural Expert) ---
    prompts_ham = [get_prompt(subject=s, message=m, label=ENRON_LABEL_INDEX_MAP.inv[0], max_seq_length=args.max_seq_len) for s, m in zip(subjects, messages)]
    prompts_spam = [get_prompt(subject=s, message=m, label=ENRON_LABEL_INDEX_MAP.inv[1], max_seq_length=args.max_seq_len) for s, m in zip(subjects, messages)]
    
    prompts = prompts_ham + prompts_spam
    
    with torch.no_grad():
        llm_log_probs = get_seq_log_prob(prompts, tokenizer, model, device)
        
    # Reshape: [Batch, 2] where dim 1 is (Ham, Spam)
    llm_scores = rearrange(llm_log_probs, '(c b) -> b c', c=2)

    # --- 2. Get N-gram Scores (Statistical Expert) ---
    ngram_scores = []
    for subj, msg in zip(subjects, messages):
        text = f"{subj} {msg}"
        score_ham = ham_expert.get_log_prob(text)
        score_spam = spam_expert.get_log_prob(text)
        ngram_scores.append([score_ham, score_spam])
    
    ngram_scores = torch.tensor(ngram_scores, device=device)

    # --- 3. Combine (wPoE) ---
    # Formula: Final = alpha * LLM + (1-alpha) * Ngram
    # Note: Since we are in log-space, Product becomes Sum.
    alpha = args.wpoe_alpha
    
    # Normalize scores roughly to same scale (optional but helpful)
    # Here we just apply the weighting directly as per the paper's concept
    final_scores = (alpha * llm_scores) + ((1.0 - alpha) * ngram_scores)

    probs = F.softmax(final_scores, dim=-1)
    return probs.detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--wpoe_alpha", type=float, default=0.8, help="Weight for LLM (0.0-1.0). 1.0 = LLM only.")
    parser.add_argument("--training_data", type=str, default="autograder/cpen455_released_datasets/train_val_subset.csv")
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--checkpoint_path", type=str, default="examples/ckpts/model_seed1.pt", help="Path to finetuned checkpoint (optional)")
    args = parser.parse_args()

    device = set_device()
    
    # --- Load LLM ---
    load_dotenv()
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)
    model = LlamaModel(config)
    
    # Try to load your BEST finetuned checkpoint if available, otherwise base
    ckpt_path = args.checkpoint_path
    if os.path.exists(ckpt_path):
        print(f"Loading Finetuned Checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    else:
        print("Loading Base Model (Zero-Shot)")
        load_model_weights(model, checkpoint, cache_dir=model_cache_dir, device=device)
    
    model = model.to(device)
    model.eval()

    # --- Train N-gram Experts ---
    print("Training N-gram Experts on Training Data...")
    df = pd.read_csv(args.training_data)
    # Filter Spam and Ham
    # Note: Spam/Ham column contains "1" for spam and "0" for ham as strings
    spam_mask = df['Spam/Ham'] == '1'
    ham_mask = df['Spam/Ham'] == '0'
    
    spam_texts = (df[spam_mask]['Subject'].astype(str) + " " + df[spam_mask]['Message'].astype(str)).tolist()
    ham_texts = (df[ham_mask]['Subject'].astype(str) + " " + df[ham_mask]['Message'].astype(str)).tolist()
    
    spam_expert = NgramExpert()
    spam_expert.train(spam_texts)
    print(f"Spam expert trained on {len(spam_texts)} examples, vocab size: {len(spam_expert.vocab)}, total tokens: {spam_expert.total_count}")
    
    ham_expert = NgramExpert()
    ham_expert.train(ham_texts)
    print(f"Ham expert trained on {len(ham_texts)} examples, vocab size: {len(ham_expert.vocab)}, total tokens: {ham_expert.total_count}")
    print("Experts Trained!")

    # --- Run Inference ---
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(args.prob_output_folder):
        os.makedirs(args.prob_output_folder)
    
    save_path = os.path.join(os.getcwd(), f"{args.prob_output_folder}/test_dataset_probs.csv")
    if os.path.exists(save_path):
        os.remove(save_path)

    print(f"Running wPoE Inference (Alpha={args.wpoe_alpha})...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Processing batches"):
            probs = wpoe_classifier(args, model, batch, tokenizer, device, spam_expert, ham_expert)
            
            data_index, _, _, _ = batch
            indices = torch.as_tensor(data_index).view(-1).tolist()
            rows = zip(indices, probs[:, 0].tolist(), probs[:, 1].tolist())
            
            file_exists = os.path.exists(save_path)
            with open(save_path, "a", newline="") as handle:
                if not file_exists:
                    handle.write("data_index,prob_ham,prob_spam\n")
                handle.writelines(f"{idx},{ham},{spam}\n" for idx, ham, spam in rows)
    
    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    main()

