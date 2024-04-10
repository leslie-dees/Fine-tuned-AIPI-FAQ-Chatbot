import os
import argparse
from utils import build_prompts

def parse_args():
    parser = argparse.ArgumentParser(description='CLI')
    parser.add_argument('--num', type=int, required=True, help='number of prompts to generate')
    return parser.parse_args()

def main():
    # Config
    args = parse_args()
    dataset = "databricks/databricks-dolly-15k"

    # Build prompts
    if not os.path.exists("./data"):
        print(f"Making dir ./data")
        os.makedirs("./data") 
    _ = build_prompts(dataset, args.num)

if __name__ == '__main__':
    main()