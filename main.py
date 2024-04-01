import argparse
import numpy as np
import torch
import sys 
from config import GemmaConfig, get_config_for_7b, get_config_for_2b
from tokenizer import Tokenizer
import contextlib
import os
from model import GemmaForCausalLM


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main(args):
    VARIANT = "7b-it"
    weights_dir = "gemma-ckpt"
    model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    if not os.path.exists(weights_dir):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it-pytorch", token = "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")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it-pytorch", token = "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")
        ckpt_path = model.state_dict()
    else:    
        model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
        model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
        
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
        model.load_weights(ckpt_path)
        model = model.to(device).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "7b"])
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--output_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    args = parser.parse_args()

    main(args)
