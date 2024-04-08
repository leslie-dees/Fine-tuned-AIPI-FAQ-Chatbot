import os
import wandb
import ollama
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger

from datasets import load_dataset

## ---------------------------------
#                CONSTANTS
## ---------------------------------

SYSTEM_PROMPT = """
  You are a helpful AI assistant. Users will ask you questions. 
  Take a moment to think then respond with a polite and 
  appropriate answer. You may use the provided CONTEXT if it 
  is useful and improves your response. If you are unsure of the 
  answer, you can respond "I don't know." or "I'm not sure.".

  CONTEXT: {context}
"""

## ---------------------------------
#                FXNS
## ---------------------------------

def init_wandb(project_name):
    # Initialise wandb.config via Pytorch-Lightning
    logger = WandbLogger(project=project_name, name=None)
    _ = logger.experiment.config
    _ = logger.experiment.path

    # Update run name
    timestamp = datetime.now().strftime('%y%m%d-%H%M')
    run_name = f"{timestamp}-A{wandb.config.alpha}-D{wandb.config.dropout}-R{wandb.config.rank}-S{wandb.config.max_seq}"
    wandb.run.name = run_name
    wandb.config['run_name'] = run_name

    # Create output dir
    output_dir = os.path.join("./adapters", run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Adapter directory created: {output_dir}')
        wandb.config['adapter_dir'] = output_dir
        
        print("WandB Logger Initialized")
        return logger, output_dir
    else:
        raise Exception("Error: Failed to initialise wandb.config")

def create_prompt(sample):
  return {
    "messages": [
      {"role": "system", "content": SYSTEM_PROMPT.format(context=sample["context"])},
      {"role": "user", "content": sample["instruction"]},
      {"role": "assistant", "content": sample["response"]}
    ]
  }

def build_prompts(dataset_id, num):

    print(f"Building {num} prompts from {dataset_id}...")
    # Load dataset from the hub
    dataset = load_dataset(dataset_id)

    # Shuffle and convert to ChatML prompt format
    prompts = dataset['train'].shuffle().select(range(num)).map(create_prompt, remove_columns=dataset['train'].features, batched=False)

    # Split to train/val & save prompts to disk
    split_prompts = prompts.train_test_split(test_size=0.2)
    split_prompts['train'].to_json("data/train_prompts.json", orient="records")
    split_prompts['test'].to_json("data/test_prompts.json", orient="records")
    print("train/test prompts saved to ./data")
  
def get_preds(test_prompts, pipe):
    preds = []
    for sample in tqdm(test_prompts.select(range(test_prompts.num_rows))):
      prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
      outputs = pipe(prompt, max_new_tokens=256, temperature=0.1, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
      sample_pred = outputs[0]['generated_text'][len(prompt):].strip()
      preds.append(sample_pred)
    return preds

def get_labels(test_prompts):
    labels = []
    for sample in tqdm(test_prompts.select(range(test_prompts.num_rows))):
      sample_label = sample["messages"][2]["content"]
      labels.append(sample_label)
    return labels

def evaluate(preds, targets):
  eval_prompt = """
  Given Answer A and Answer B, decide if they are similar or different. 
  Your must strictly give a single word response, either 'similar' or 'different'.
  Do not provide any additional justification.

  Answer A: {answer_a}
  
  Answer B: {answer_b}
  """
  responses = []
  for ids, (pred, target) in enumerate(zip(preds,targets)):
    prompt = eval_prompt.format(answer_a=pred, answer_b=target)
    message = ollama.generate(model='llama2', prompt=prompt)
    response = message['response'].lower().strip()
    print(response)
    if 'similar' in response:
      responses.append("similar")
    elif 'different' in response:
      responses.append("different")
    else:
      responses.append("n/a")
  
  return responses
    