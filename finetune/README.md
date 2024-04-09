## Chatbot Finetuning Pipeline

This pipeline is used to finetune [`Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset

## Setup

**Developer Environment**

```
$ sudo apt-get install python3-venv
$ python3 -m venv ~/.venv
$ source ~/.venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

**Enable Flash Attention**

```
!MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**WandB Login**
```
# Create account at https://wandb.ai/ > Init new project > Copy API key
$ wandb login # paste API key
```

**HuggingFace Hub Login**
```
# Create a HF Access Token with Write permissions > Copy Token
$ huggingface-cli login # paste token
```

**Install Ollama**
```
$ curl -fsSL https://ollama.com/install.sh | sh
$ ollama pull llama2
```

## Finetuning w/ PEFT QLoRA

```
# Activate environment
$ source ~/.venv/bin/activate
```

### 1. QLoRA Config

Configure model and runtime parameters in [config-defaults.yaml](config-defaults.yaml) 

### 2. Build Prompts

Build training/validation prompt sets from `databricks/databricks-dolly-15k`

```
$ python3 prompts.py --num <num_prompts>
```

### 3. Run Finetuning

```
# Gotchas: Manage CUDA memory
$ rm -rf ../.cache/huggingface/datasets

# Run finetuning
$ python3 finetune.py --project <wandb_project_name>
# i.e. python3 finetune.py --project mistral-bot
```

After finetuning, to merge the adapter with the base model, run the [`merge.py`](./merge.py) script passing the HuggingFace adapter ID.

WARNING: Merge requires > 16GB memory

```
$ python3 merge.py --id <adapter_id>
# i.e. python3 merge.py --id kahliahogg/mistral-bot
```

### 4. Inference

An inference playground is available in the [`inference.ipynb`](./inference.ipynb) notebook

### 5. Evaluation

Evaluation is performed using an LLM-as-judge in the [`eval.ipynb`](./eval.ipynb) notebook