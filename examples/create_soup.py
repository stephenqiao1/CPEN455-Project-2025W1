import torch
import os
import argparse

def create_soup(model_paths, output_path):
    """
    Loads multiple checkpoints and averages their state_dicts
    """
    if not model_paths:
        print("No models provided!")
        return

    # Load the first model to serve as the base
    print(f"Loading base model: {model_paths[0]}")
    soup_state_dict = torch.load(model_paths[0], map_location="cpu")

    # Iterate through the rest and add their weights
    for i, path in enumerate(model_paths[1:], start=1):
        print(f"Adding model {i}: {path}")
        current_state_dict = torch.load(path, map_location="cpu")

        for key in soup_state_dict:
            soup_state_dict[key] += current_state_dict[key]

    # Divide by the number of models to get the average
    num_models = len(model_paths)
    print(f"Averaging weights across {num_models} models...")
    for key in soup_state_dict:
        soup_state_dict[key] = soup_state_dict[key] / num_models

    # Save the fresh soup
    print(f"Saving Model Soup to: {output_path}")
    torch.save(soup_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    ingredients = [
        "examples/ckpts/model_seed1.pt",
        "examples/ckpts/model_seed2.pt",
        "examples/ckpts/model_seed3.pt",
    ]

    create_soup(ingredients, "examples/ckpts/model_soup.pt")