import os
import torch
import random
import argparse
import warnings

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='CLI')
    parser.add_argument('--id', type=str, required=True, help='HF PEFT Adapter ID')
    return parser.parse_args()

def main():
    # Free CUDA memory
    torch.cuda.empty_cache()

    args = parse_args()
    adapter_id = args.id

    # Output directory
    output_dir = "./model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Output directory created: {output_dir}')

    # GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")

    # Load PEFT adapter
    print(f"Loading adapter {args.id}")
    adapter = AutoPeftModelForCausalLM.from_pretrained(
        adapter_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_id)

    # Merge adapter w/ base model and save
    print("Merging adapter w/ base model")
    model = adapter.merge_and_unload()
    model.save_pretrained(f"{output_dir}", safe_serialization=True, max_shard_size="2GB")
    print(f"Model saved to: {output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    print(f"Tokenizer saved to: {output_dir}")

if __name__ == '__main__':
    main()