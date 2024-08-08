import bitsandbytes
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import pandas as pd
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Seq2SeqTrainingArguments, \
    BitsAndBytesConfig, set_seed, Seq2SeqTrainer
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from adapters import AdapterTrainer, LoRAConfig, LlamaAdapterModel, Seq2SeqAdapterTrainer, AutoAdapterModel, \
    AdapterConfig, SeqBnConfig, DoubleSeqBnConfig, DoubleSeqBnInvConfig, ParBnConfig, IA3Config
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

try:
  from google.colab import drive
  drive.mount('/content/drive')
  dir = "/content/drive/My Drive/Colab Notebooks/"
except Exception as e:
  print("Exceptoion: ", e)
  dir = ""

import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

accelerator = Accelerator()
torch.cuda.empty_cache()


def loading_dataset(dataset_name, force_run=False):
    local_path = dir + dataset_info_dict[dataset_name]["local_path"]
    if not os.path.exists(local_path) or force_run:
        dataset_id = dataset_info_dict[dataset_name]["dataset_id"]
        dataset = load_dataset(path=dataset_id, streaming=True, trust_remote_code=True)
        dataset_dict = {}
        for split, data in dataset.items():
            # if split == "train":
            data = list(data)[:500] # TODO: selecting random 1k or 5k or 10k
            df = pd.DataFrame(data)
            dataset_dict[split] = Dataset.from_pandas(df) # dataset[split].to_pandas())
        _dataset = DatasetDict(dataset_dict)
        _dataset.save_to_disk(local_path)
        return _dataset
    else:
        dataset = load_from_disk(local_path)
    return dataset


def preprocess_dataset(_data):
    return {"text": _data["article"], "summary": _data["abstract"]}

"""## Configs"""

dataset_info_dict = {
    "arxiv": {
        "dataset_id": "ccdv/arxiv-summarization",
        "local_path": "domains/arxiv_summarization"
    },
    "pubmed": {
        "dataset_id": "ccdv/pubmed-summarization",
        "local_path": "domains/pubmed_summarization"
    }
}
max_seq_len = 256 # context window # 1024
ddtype = torch.float32 # bfloat16
bits = 4 #8
bf16 = False
bf32 = False
fp16 = True
compute_dtype = torch.float32 # torch.bfloat16 if bf16 else torch.float32

# Model Config
llama31 = "meta-llama/Meta-Llama-3.1-8B-Instruct" # works only with transformers==4.43.3
llama3 = "meta-llama/Meta-Llama-3-8B-Instruct"
llama2 = "meta-llama/Llama-2-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device: ", device)

if device == "cuda":
  n_gpus = torch.cuda.device_count()
  print("gpus-> ", n_gpus)

from huggingface_hub import login
login(token="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")

"""# LLama Model"""

model_id = llama3 # llama2

# Load 4-bit quantized model
model = LlamaAdapterModel.from_pretrained( # AutoAdapterModel
# model = AutoModelForCausalLM.from_pretrained(   #  LlamaAdapterModel.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    ),
    torch_dtype=ddtype,
)
model.config.use_cache = False
# setattr(model, 'model_parallel', True)
# setattr(model, 'is_parallelizable', True)


# def find_all_linear_names(bits, model):
clsi = bitsandbytes.nn.Linear4bit if bits == 4 else (bitsandbytes.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
lora_module_names = set()
for name, module in model.named_modules():
    if isinstance(module, clsi):
        names = name.split('.')
        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

if 'lm_head' in lora_module_names: # needed for 16-bit
    lora_module_names.remove('lm_head')
"""# Tokenizer"""

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", tokenizer_type="llama",
                                          trust_remote_code=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
})


def tokenization_process(input_data):
    input_data.pop("article", None)
    input_data.pop("abstract", None)
    text = input_data.get('text', "")
    summary = input_data.get('summary', "")

    inputs = tokenizer(text, max_length=max_seq_len, truncation=True, padding="max_length",
                       return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(summary, max_length=max_seq_len, truncation=True, padding="max_length",
                            return_tensors="pt")
    # For causal LM, concatenate the input and output for training
    input_ids = inputs['input_ids'].clone()
    labels = targets['input_ids'].clone()
    return {"input_ids": input_ids, "attention_mask": inputs['attention_mask'].clone(), "labels": labels}


model.resize_token_embeddings(len(tokenizer))
model.config.torch_dtype = torch.float32
model.config.torch_dtype=torch.float32 # (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float16))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


# Adapters for the different dataset summarization tasks

lora_config = LoRAConfig( # for pubmed
    selfattn_lora=True, intermediate_lora=True, output_lora=True,
    attn_matrices=["q", "k", "v"],
    alpha=16, r=64, dropout=0.1,
)

ia3_config = IA3Config( # for arxiv
    selfattn_lora=True, intermediate_lora=True, output_lora=True,
    attn_matrices=["q", "k", "v"],
    alpha=1, r=1, dropout=0.1,
)


peft_pubmed = "lora_pubmed"
peft_arxiv = "ia3_arxiv"

"""  Adding pubmed peft to model  """

print("** Adding pubmed peft to model **")

model.add_adapter(peft_pubmed, config=lora_config)
model.add_causal_lm_head(peft_pubmed)

model.set_active_adapters(peft_pubmed)
model.train_adapter(peft_pubmed)
model.adapter_to(peft_pubmed, device=device)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

print("With PubMed:\n", model.adapter_summary())

for param in model.parameters():
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

print("PubMed Model: ", model)


""" Loading both datasets """
print("**** Loading PubMed **** ")
loaded_dataset = loading_dataset("pubmed")
loaded_dataset = loaded_dataset.map(preprocess_dataset, batched=True)

pubmed_data = {
    "train": loaded_dataset["train"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
    "val": loaded_dataset["validation"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
    "test": loaded_dataset["test"].map(tokenization_process, batched=True)
}

print("\n**** Loading Arxiv **** ")
loaded_dataset = loading_dataset("arxiv")
loaded_dataset = loaded_dataset.map(preprocess_dataset, batched=True)

arxiv_data = {
    "train": loaded_dataset["train"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
    "val": loaded_dataset["validation"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
    "test": loaded_dataset["test"].map(tokenization_process, batched=True)
}

del loaded_dataset


print("** Training model with pubmed peft **")

set_seed(42)
torch.enable_grad()
training_args = TrainingArguments( # Seq2Seq
    remove_unused_columns=False,
    output_dir="results/fusion_exp",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    logging_steps=10,
    save_steps=500,
    eval_steps=187,
    num_train_epochs=1,
    save_total_limit=3,
    gradient_accumulation_steps=16,
    max_steps=30,
    lr_scheduler_type="constant",
    optim="adamw_hf",
    learning_rate=0.0002,
    group_by_length=True,
    bf16=True,
    warmup_ratio=0.03,
    max_grad_norm=0.3
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_seq_len, label_pad_token_id=-100)
trainer = AdapterTrainer( # AdapterTrainer # Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=pubmed_data["train"],
    eval_dataset=pubmed_data["val"],
    args=training_args,
)

trainer = accelerator.prepare(trainer)
trainer.train()

results = trainer.evaluate()
print("Results from Training PubMed: \n", results)

"""# Save Adapter """

adapter_save_dir = "saved_models/"
# model.save_pretrained("{}/arxiv_lora_adapter".format(adapter_save_dir))
# model.save_adapter(adapter_save_dir+"arxiv_lora_adapter_aml", "arxiv_adapter")
model.save_adapter(adapter_save_dir+"fusion_experiment", peft_pubmed)

print("** Training with PEFT for PubMed Compelete and Adapter saved. ")


"""  Adding arxiv peft to model  """

print("\n\n\n** Adding Arxiv peft to model **")

model.add_adapter(arxiv_data, config=ia3_config)
model.add_causal_lm_head(arxiv_data)

model.set_active_adapters(arxiv_data)
model.train_adapter(arxiv_data)
model.adapter_to(arxiv_data, device=device)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

print("With Arxiv:\n", model.adapter_summary())


print("** Training model with arxiv peft **")

set_seed(42)
torch.enable_grad()
# training_args = TrainingArguments( # Seq2Seq
#     remove_unused_columns=False,
#     output_dir="results/fusion_exp",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=2,
#     evaluation_strategy="steps",
#     logging_steps=10,
#     save_steps=500,
#     eval_steps=187,
#     num_train_epochs=1,
#     save_total_limit=3,
#     gradient_accumulation_steps=16,
#     max_steps=30,
#     lr_scheduler_type="constant",
#     optim="adamw_hf",
#     learning_rate=0.0002,
#     group_by_length=True,
#     bf16=True,
#     warmup_ratio=0.03,
#     max_grad_norm=0.3
# )

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_seq_len, label_pad_token_id=-100)
trainer = AdapterTrainer( # AdapterTrainer # Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=arxiv_data["train"],
    eval_dataset=arxiv_data["val"],
    args=training_args,
)

trainer = accelerator.prepare(trainer)
trainer.train()

results = trainer.evaluate()
print("Results from Training Arxiv: \n", results)

"""# Save Adapter """

# model.save_pretrained("{}/arxiv_lora_adapter".format(adapter_save_dir))
# model.save_adapter(adapter_save_dir+"arxiv_lora_adapter_aml", "arxiv_adapter")
model.save_adapter(adapter_save_dir+"fusion_experiment", peft_arxiv)

print("** Training with PEFT for Arxiv Compelete and Adapter saved. ")


""" *** Testing for Both simultaneuosly *** """