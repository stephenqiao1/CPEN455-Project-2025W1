import torch
import torch.nn as nn
import math

class LoraLinear(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze the base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False 
        
        # Create LoRA matrices (A and B)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        # Initialize weights (A=Gaussian, B=Zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original path (Frozen)
        base_output = self.base_layer(x)

        # LoRA path (Trainable)
        # W + BA -> Output = Wx + BAx
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        
        return base_output + lora_output
    
def apply_lora_to_model(model, r=8, lora_alpha=16):
    """
    Applies LoRA to all linear layers in the transformer. 
    """
    print(f"Applying LoRA with rank {r} and alpha {lora_alpha}...")

    for layer in model.layers:
        # Target the Self-Attention block
        attn = layer.self_attn
        attn.q_proj = LoraLinear(attn.q_proj, r=r, lora_alpha=lora_alpha)
        attn.k_proj = LoraLinear(attn.k_proj, r=r, lora_alpha=lora_alpha)
        attn.v_proj = LoraLinear(attn.v_proj, r=r, lora_alpha=lora_alpha)
        attn.o_proj = LoraLinear(attn.o_proj, r=r, lora_alpha=lora_alpha)

        # MLP Block 
        mlp = layer.mlp
        mlp.gate_proj = LoraLinear(mlp.gate_proj, r=r, lora_alpha=lora_alpha)
        mlp.up_proj = LoraLinear(mlp.up_proj, r=r, lora_alpha=lora_alpha)
        mlp.down_proj = LoraLinear(mlp.down_proj, r=r, lora_alpha=lora_alpha)   

    # Mark only LoRA parameters as trainable
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"LoRA applied. Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

    return model