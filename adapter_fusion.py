import adapters
import bitsandbytes
import evaluate
import nltk
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import pandas as pd
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Seq2SeqTrainingArguments, \
    BitsAndBytesConfig, set_seed, Seq2SeqTrainer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from adapters import AdapterTrainer, LoRAConfig, LlamaAdapterModel, Seq2SeqAdapterTrainer, AutoAdapterModel, \
    AdapterConfig, SeqBnConfig, DoubleSeqBnConfig, DoubleSeqBnInvConfig, ParBnConfig, IA3Config
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from trl import SFTTrainer

try:
    from google.colab import drive

    drive.mount('/content/drive')
    dir = "/content/drive/My Drive/Colab Notebooks/"
except Exception as e:
    print("Exceptoion: ", e)
    dir = ""

import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
            data = list(data)[:500]  # TODO: selecting random 1k or 5k or 10k
            df = pd.DataFrame(data)
            dataset_dict[split] = Dataset.from_pandas(df)  # dataset[split].to_pandas())
        _dataset = DatasetDict(dataset_dict)
        _dataset.save_to_disk(local_path)
        return _dataset
    else:
        dataset = load_from_disk(local_path)
    return dataset


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

DEFAULT_SYSTEM_PROMPT = """
    Given below is an article as input from some dataset. Write a  concise and informative summary as a Response for the article.
""".strip()

def generate_training_prompt(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    prompt = """### Instruction: {}\n\n### Input: {}\n\n### Response: {}""".format(system_prompt, article, summary)

    return prompt.strip()


def preprocess_dataset(_sample):
    texts = [generate_training_prompt(article=article, summary=summary)
             for article, summary in zip(_sample["article"], _sample["abstract"])]

    return {
        "content": _sample["article"],
        "summary": _sample["abstract"],
        "text": texts
    }

    # return {"text": _data["article"], "summary": _data["abstract"]}

# def process_tokenization_with_prompt(input_data):
#     input_data.pop("article", None)
#     input_data.pop("abstract", None)
#     inputs = [f"Summarize the following text:\n\n{abstract}" for abstract in input_data["text"]]
#     model_inputs = tokenizer(inputs, max_length=max_seq_len, truncation=True)
#
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(input_data["summary"], max_length=max_seq_len, truncation=True)
#
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


def process_dataset_with_prompt(_dataset: Dataset):
    return (_dataset.map(preprocess_dataset, batched=True, remove_columns=["article", "abstract"]
                         ).shuffle(seed=42))
    # return data


max_seq_len = 1024  # context window # 1024
ddtype = torch.bfloat16  # bfloat16
bits = 4  # 8
bf16 = False
bf32 = False
fp16 = True
compute_dtype = torch.bfloat16  # torch.bfloat16 if bf16 else torch.float32

# Model Config
llama31 = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # works only with transformers==4.43.3
llama3 = "meta-llama/Meta-Llama-3-8B-Instruct"
llama2 = "meta-llama/Llama-2-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device: ", device)


""" Loading both datasets """
print("**** Loading PubMed **** ")
loaded_dataset = loading_dataset("arxiv")
# loaded_dataset = loaded_dataset.map(preprocess_dataset, batched=True)

# pubmed_data = {
#     "train": loaded_dataset["train"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
#     "val": loaded_dataset["validation"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
#     "test": loaded_dataset["test"].map(tokenization_process, batched=True)
# }

print("Loaded Dataset: ", loaded_dataset)

pubmed_data = {
    "train": process_dataset_with_prompt(loaded_dataset["train"]), #.map(process_tokenization_with_prompt, batched=True,
                                         # remove_columns=["text", "summary"]),
    "val": process_dataset_with_prompt(loaded_dataset["validation"]), # .map(process_tokenization_with_prompt, batched=True,
                                            #remove_columns=["text", "summary"]),
    "test": process_dataset_with_prompt(loaded_dataset["test"]) #.map(process_tokenization_with_prompt, batched=True)
}


print("PUBMED: ", pubmed_data)
# print("\n**** Loading Arxiv **** ")
# loaded_dataset = loading_dataset("arxiv")
# loaded_dataset = loaded_dataset.map(preprocess_dataset, batched=True)
#
# arxiv_data = {
#     "train": loaded_dataset["train"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
#     "val": loaded_dataset["validation"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
#     "test": loaded_dataset["test"].map(tokenization_process, batched=True)
# }

if device == "cuda":
    n_gpus = torch.cuda.device_count()
    print("gpus-> ", n_gpus)

from huggingface_hub import login

login(token="")

"""# LLama Model"""

model_id = llama3  # llama2

# Load 4-bit quantized model
# model = LlamaAdapterModel.from_pretrained(  # AutoAdapterModel
model = AutoAdapterModel.from_pretrained(   #  AutoModelForCausalLM.from_pretrained(
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
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
setattr(model, 'model_parallel', True)
setattr(model, 'is_parallelizable', True)

# def find_all_linear_names(bits, model):
clsi = bitsandbytes.nn.Linear4bit if bits == 4 else (bitsandbytes.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
lora_module_names = set()
for name, module in model.named_modules():
    if isinstance(module, clsi):
        names = name.split('.')
        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

if 'lm_head' in lora_module_names:  # needed for 16-bit
    lora_module_names.remove('lm_head')
"""# Tokenizer"""

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", tokenizer_type="llama",
                                          trust_remote_code=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
})

# model.resize_token_embeddings(len(tokenizer))
model.config.torch_dtype = torch.bfloat16  # (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float16))


def tokenization_process(input_data):
    text = input_data.pop('text', "")
    summary = input_data.pop('summary', "")
    input_data.pop("content")

    inputs = tokenizer(text, max_length=max_seq_len, truncation=True, padding="max_length",
                       return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(summary, max_length=max_seq_len, truncation=True, padding="max_length",
                            return_tensors="pt")
    # For causal LM, concatenate the input and output for training
    input_ids = inputs["input_ids"].squeeze()
    labels = targets["input_ids"].squeeze()
    # input_ids = inputs['input_ids'].clone()
    # labels = targets['input_ids'].clone()
    return {"input_ids": input_ids, "labels": labels}


# def tokenization_process(input_data):
#     text = input_data["text"]
#     summary = input_data["summary"]
#
#     inputs = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
#     # with tokenizer.as_target_tokenizer():
#     #     targets = tokenizer(summary, max_length=150, truncation=True, padding="max_length", return_tensors="pt")
#     #
#     # input_ids = inputs["input_ids"].squeeze()
#     # labels = targets["input_ids"].squeeze()
#
#     return {
#         "input_ids": inputs["input_ids"].squeeze(),
#         "attention_mask": inputs["attention_mask"].squeeze(),
#         "labels": inputs["input_ids"].squeeze()  # This is for causal LM where labels are the input ids shifted by 1
#     }


pubmed_data["train"] = pubmed_data["train"].map(tokenization_process, batched=True)
pubmed_data["val"] = pubmed_data["val"].map(tokenization_process, batched=True)
# pubmed_data["test"] = pubmed_data["test"].map(tokenization_process, batched=True)

# Adapters for the different dataset summarization tasks
adapters.init(model)

lora_config = LoRAConfig(  # for pubmed
    selfattn_lora=True, intermediate_lora=True, output_lora=True,
    attn_matrices=["q", "k", "v"],
    alpha=16, r=64, dropout=0.1,
)

ia3_config = IA3Config(  # for arxiv
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

# model.add_adapter(peft_arxiv, config=ia3_config)
# model.add_causal_lm_head(peft_arxiv)


model.set_active_adapters([peft_pubmed])
model.train_adapter([peft_pubmed])
model.adapter_to(peft_pubmed, device=device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name} - {param.shape}")
print("\nWith PubMed:\n", model.adapter_summary())

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

for param in model.parameters():
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
print("PubMed Model: ", model)

print("Qconfig: ", model.config.quantization_config)

# class CastOutputToFloat(torch.nn.Sequential):
#     def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)

rouge = evaluate.load("rouge")


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels since -100 is used to ignore padding in labels
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Extract ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return result

print("** Training model with pubmed peft **")

torch.enable_grad()
training_args = TrainingArguments(  # Seq2Seq
    remove_unused_columns=False,
    output_dir="results/pubmed_lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    eval_strategy="epoch",
    logging_steps=10,
    save_steps=10,
    seed=42,
    eval_steps=10,
    num_train_epochs=1,
    gradient_accumulation_steps=10,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=5e-3,
    group_by_length=True,
    bf16=True,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_seq_len,
                                       label_pad_token_id=-100)
trainer = AdapterTrainer(  # AdapterTrainer # Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=pubmed_data["train"],
    eval_dataset=pubmed_data["val"],
    args=training_args,
    # compute_metrics=compute_metrics
)

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     dataset_text_field="text",
#     train_dataset=pubmed_data["train"],
#     eval_dataset=pubmed_data["val"],
#     args=training_args,
# )

accelerator.prepare(trainer)
trainer.train()

results = trainer.evaluate()
print("Results from Training PubMed: \n", results)

ft_model = trainer.model
#
# """# Save Adapter """
adapter_save_dir = "saved_models/"
# model.save_pretrained("{}/arxiv_lora_adapter".format(adapter_save_dir))
# model.save_adapter(adapter_save_dir+"arxiv_lora_adapter_aml", "arxiv_adapter")
ft_model.save_adapter(adapter_save_dir + peft_pubmed, peft_pubmed)
ft_model.merge_adapter(peft_pubmed)


def inference_prompt(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    prompt = """
        ### Instruction:
            {}
            
        ### Input:
            {}
            
        ### Response:
    """.format(system_prompt.strip(), article.strip())

    return prompt.strip()



def generate_summary(model, text, truth):
    # prompt = f"Summarize the following text:\n\n{abstract}"
    text = inference_prompt(text)
    print("Text: \n", text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    in_len = len(inputs["input_ids"][0])
    with torch.inference_mode():
        summary_ids = model.generate(**inputs,
                                     # max_length=512, # do_sample=True,  # Enable sampling
                                     top_k=50,  # Top-k sampling
                                     num_return_sequences=1,  # Generate a single sequence
                                     early_stopping=True,
                                     # temprature=0.001,
                                     max_new_tokens=150)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Truth:\n{}\n\n\nPrediction:\n{} ".format(truth, summary))

    print("\n\n\nRouge Scores: ", rouge.compute(references=[truth], predictions=[summary]))
    # return summary


test_data = pd.DataFrame(pubmed_data["test"][:5])

sample = test_data.iloc[3]

generate_summary(model=ft_model, text=sample.content, truth=sample.summary)

# print("** Training with PEFT for PubMed Compelete and Adapter saved. ")

# """  Adding arxiv peft to model  """
#
# print("\n\n\n** Adding Arxiv peft to model **")
#
# # model.set_active_adapters(None)
# # model.train_adapter(None)
# model.set_active_adapters(peft_arxiv)
# model.train_adapter(peft_arxiv)
# model.adapter_to(peft_arxiv, device=device)
#
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
#
# print("With Arxiv:\n", model.adapter_summary())
#
# print("** Training model with arxiv peft **")
#
# set_seed(42)
# torch.enable_grad()
#
# trainer = AdapterTrainer(  # AdapterTrainer # Seq2SeqTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     train_dataset=arxiv_data["train"],
#     eval_dataset=arxiv_data["val"],
#     args=training_args,
#     # compute_metrics=compute_metrics
# )
#
# trainer = accelerator.prepare(trainer)
# trainer.train()
#
# results = trainer.evaluate()
# print("Results from Training Arxiv: \n", results)
#
# """# Save Adapter """
#
# # model.save_pretrained("{}/arxiv_lora_adapter".format(adapter_save_dir))
# # model.save_adapter(adapter_save_dir+"arxiv_lora_adapter_aml", "arxiv_adapter")
# model.save_adapter(adapter_save_dir + peft_arxiv, peft_arxiv)
# model.merge_adapter(peft_arxiv)
#
# print("** Training with PEFT for Arxiv Compelete and Adapter saved. ")
#
# """ *** Testing for Both simultaneuosly *** """