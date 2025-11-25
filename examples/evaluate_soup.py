#!/usr/bin/env python3
import os
import argparse
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from autograder.dataset import CPEN455_2025_W1_Dataset
from model import LlamaModel
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from examples.bayes_inverse import save_probs

if __name__ == "__main__":
    # Random seed for reproducibility
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--soup_path", type=str, default="examples/ckpts/model_soup.pt", help="Path to the local checkpoint file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    args = parser.parse_args()

    load_dotenv()
    
    # Standard setup
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    device = set_device()

    # Load Tokenizer & Config from the base model (Standard)
    print(f"Loading config/tokenizer from {checkpoint}...")
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Initialize Model Structure
    model = LlamaModel(config)

    # --- CRITICAL CHANGE: Load Local Soup Weights ---
    print(f"Loading local model weights from: {args.soup_path}")
    if not os.path.exists(args.soup_path):
        raise FileNotFoundError(f"Could not find checkpoint at {args.soup_path}. Did you run create_soup.py?")
        
    state_dict = torch.load(args.soup_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully!")
    # ------------------------------------------------

    model = model.to(device)
    
    # Setup Data
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Run Inference
    save_probs(args, model, tokenizer, test_dataloader, device=device, name="test")