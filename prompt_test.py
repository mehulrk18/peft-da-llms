import adapters
import wandb
import evaluate
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
    print("Exception: ", e)
    dir = ""

import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

accelerator = Accelerator()
rouge = evaluate.load("rouge")
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
    Given below is an article. Write a concise and informative Summary for the article.
""".strip()

def generate_training_prompt(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    prompt = """### Instruction: {}\n\n### Article: {}\n\n### Summary: {}""".format(system_prompt, article, summary)

    return prompt.strip()


def inference_prompt(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    prompt = """### Instruction: {}\n### Article: {}\n### Summary:""".format(system_prompt.strip(), article.strip())

    return prompt.strip()


def generate_summary(model, tokenizer, text, truth):
    # content = f"Summarize the following text:\n\n{text}"
    content = inference_prompt(text)
    # print("Text: \n", text)
    inputs = tokenizer(content, return_tensors="pt").to(device)
    in_len = len(inputs["input_ids"][0])
    with torch.inference_mode():
        summary_ids = model.generate(**inputs,
                                     # max_length=512, # do_sample=True,  # Enable sampling
                                     top_k=50,  # Top-k sampling
                                     num_return_sequences=1,  # Generate a single sequence
                                     # early_stopping=True,
                                     # temprature=0.001,
                                     max_new_tokens=150)
    summary = tokenizer.decode(summary_ids[0][in_len:], skip_special_tokens=True)

    print("Truth:\n{}\n\n\nPrediction:\n{} ".format(truth, summary))

    print("\n\n\nRouge Scores: ", rouge.compute(references=[truth], predictions=[summary]))
    # return summary


# def generate_summary(model, content):
#     # Defining the template to generate summary
#     template = """
#     Write a concise summary of the text, return your responses with 5 lines that cover the key points of the text.
#     ```{text}```
#     SUMMARY:
#     """
#     prompt = PromptTemplate(template=template, input_variables=["text"])
#     llm_chain = LLMChain(prompt=prompt, llm=model)
#
#     summary = llm_chain.run(text_chunk)
#     return summary

def preprocess_dataset(_sample):
    texts = [generate_training_prompt(article=article, summary=summary)
             for article, summary in zip(_sample["article"], _sample["abstract"])]

    return {
        # "content": _sample["article"],
        # "summary": _sample["abstract"],
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
    return (_dataset.map(preprocess_dataset, batched=True, # remove_columns=["article", "abstract"]
                         ).shuffle(seed=42))
    # return data


max_seq_len = 1024  # context window # 1024
dataset_name = "arxiv"
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
mistral = "mistralai/Mistral-7B-v0.3"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device: ", device)

model_name_dict = {
    llama3: "LLaMA3",
    llama31: "LLaMA3.1",
    llama2: "LLaMA2",
    mistral: "Mistral"
}

""" Loading both datasets """
print("**** Loading PubMed **** ")
loaded_dataset = loading_dataset(dataset_name)
# loaded_dataset = loaded_dataset.map(preprocess_dataset, batched=True)

# pubmed_data = {
#     "train": loaded_dataset["train"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
#     "val": loaded_dataset["validation"].map(tokenization_process, batched=True, remove_columns=["text", "summary"]),
#     "test": loaded_dataset["test"].map(tokenization_process, batched=True)
# }

print("\nLoaded Dataset: ", loaded_dataset)

pubmed_data = {
    "train": process_dataset_with_prompt(loaded_dataset["train"]), #.map(process_tokenization_with_prompt, batched=True,
                                         # remove_columns=["text", "summary"]),
    "val": process_dataset_with_prompt(loaded_dataset["validation"]), # .map(process_tokenization_with_prompt, batched=True,
                                            #remove_columns=["text", "summary"]),
    "test": loaded_dataset["test"] # .map(process_tokenization_with_prompt, batched=True)
}


print("\nPUBMED: ", pubmed_data)
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

login(token="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")

os.environ['WANDB_API_KEY'] = "eac936af5312d5c773d1c970723d73a4325b3bd0"
wandb.login()

"""# LLama Model"""

model_id = llama3  # llama3

# Load 4-bit quantized model
uft_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=bits == 4,
    #     load_in_8bit=bits == 8,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=compute_dtype,
    # ),
    torch_dtype=ddtype,
)

uft_model.config.use_cache = False
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
setattr(uft_model, 'model_parallel', True)
setattr(uft_model, 'is_parallelizable', True)


# model = LlamaAdapterModel.from_pretrained(  # AutoAdapterModel
model = AutoAdapterModel.from_pretrained(   #  AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=bits == 4,
    #     load_in_8bit=bits == 8,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=compute_dtype,
    # ),
    torch_dtype=ddtype,
)
model.config.use_cache = False
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
setattr(model, 'model_parallel', True)
setattr(model, 'is_parallelizable', True)

# def find_all_linear_names(bits, model):
# clsi = bitsandbytes.nn.Linear4bit if bits == 4 else (bitsandbytes.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
# lora_module_names = set()
# for name, module in model.named_modules():
#     if isinstance(module, clsi):
#         names = name.split('.')
#         lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#
# if 'lm_head' in lora_module_names:  # needed for 16-bit
#     lora_module_names.remove('lm_head')
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
uft_model.config.torch_dtype = torch.bfloat16

def tokenization_process(input_data):
    # text = input_data.pop('text', "")
    # summary = input_data.pop('summary', "")
    # input_data.pop("content")

    inputs = tokenizer(input_data["text"], max_length=max_seq_len, truncation=True, padding="max_length",
                       return_tensors="pt")
    # with tokenizer.as_target_tokenizer():
    #     targets = tokenizer(summary, max_length=max_seq_len, truncation=True, padding="max_length",
    #                         return_tensors="pt")
    # labels = targets["input_ids"].squeeze()
    # input_ids = inputs['input_ids'].clone()
    # labels = targets['input_ids'].clone()
    return {"input_ids": inputs["input_ids"]}  # , "labels": labels}


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

test_data = pd.DataFrame(pubmed_data["test"][:5])

sample = test_data.iloc[3]
print("Unfinetuned Model: ", uft_model)
print("\n\n\n*********** SUMMARY BEFORE TRAINING ***********\n\n\n")
generate_summary(model=uft_model, tokenizer=tokenizer, text=sample.article, truth=sample.abstract)
del uft_model

pubmed_data["train"] = pubmed_data["train"].map(tokenization_process, batched=True, remove_columns=["text", "article", "abstract"])
pubmed_data["val"] = pubmed_data["val"].map(tokenization_process, batched=True, remove_columns=["text", "article", "abstract"])
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
    attn_matrices=["q_proj", "k_proj", "v_proj"],
    alpha=1, r=1, # dropout=0.1,
)

peft_pubmed = "{}_{}_lora".format(model_name_dict[model_id], dataset_name)  # "lora_pubmed", (model, dataset, adapter_name)
# peft_arxiv = "ia3_arxiv"
adapter_name = peft_pubmed
# model.add_adapter(adapter_name,
#                   config=SeqBnConfig(
#                       mh_adapter=True,
#                       output_adapter=True,
#                       reduction_factor=16,
#                       non_linearity="gelu"
#                   ))


"""  Adding pubmed peft to model  """

print("** Adding pubmed peft to model **")

# model.add_adapter(peft_pubmed, config=mis_lora_config)
model.add_adapter(peft_pubmed, config=lora_config)  #lora_config)
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
print("Adapter PubMed Model: ", model)

# print("Qconfig: ", model.config.quantization_config)

# class CastOutputToFloat(torch.nn.Sequential):
#     def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)


def compute_metrics(pred):
    # predictions, labels = eval_preds
    # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #
    # # Replace -100 in labels since -100 is used to ignore padding in labels
    # labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    # # Compute ROUGE scores
    # result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #
    # # Extract ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    #
    # return result

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode the predicted and label sequences
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
    return {
        # "rouge1": result["rouge1"].mid.fmeasure,
        # "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure,
        # "rougeLsum": result["rougeLsum"].mid.fmeasure,
    }

print("** Training model with pubmed peft **")


torch.autograd.set_detect_anomaly(True)
torch.enable_grad()
training_args = TrainingArguments(  # Seq2Seq
    # remove_unused_columns=False,
    # output_dir="results/{}_{}_ia3".format(model_name_dict[model_id], dataset_name), #pubmed_lora",
    # per_device_train_batch_size=1,
    # per_device_eval_batch_size=1,
    # evaluation_strategy="epoch",
    # logging_steps=10,
    # save_steps=50,
    # eval_steps=10,
    # save_total_limit=3,
    # num_train_epochs=1,
    # gradient_accumulation_steps=10,
    # # max_steps=1875,
    # lr_scheduler_type="constant",
    # optim="paged_adamw_32bit",
    # learning_rate=0.0002,
    # group_by_length=True,
    # bf16=True,
    # warmup_ratio=0.03,
    # report_to="tensorboard"

    remove_unused_columns=False,
    output_dir="results/{}_{}_lora_seq2seq".format(model_name_dict[model_id], dataset_name), #pubmed_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=7, # 7
    evaluation_strategy="epoch",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    logging_dir=dir+"logs/",
    # output_dir="results/",
    report_to="wandb",
    save_safetensors=True,
    lr_scheduler_type="constant",  # "cosine",
    seed=42,
    load_best_model_at_end=True,
    # push_to_hub=True,
)

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_seq_len,
#                                        label_pad_token_id=-100)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = AdapterTrainer(  # AdapterTrainer # Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=pubmed_data["train"],
    eval_dataset=pubmed_data["val"],
    args=training_args,
    # compute_metrics=compute_metrics
)

accelerator.prepare(trainer)
trainer_stats = trainer.train()

results = trainer.evaluate()
print("Results from Training PubMed: \n", results)

train_loss = trainer_stats.training_loss
print(f"Training loss:{train_loss}")


ft_model = trainer.model
#
# """# Save Adapter """
adapter_save_dir = "saved_models/"
# model.save_pretrained("{}/arxiv_lora_adapter".format(adapter_save_dir))
# model.save_adapter(adapter_save_dir+"arxiv_lora_adapter_aml", "arxiv_adapter")
ft_model.save_adapter(adapter_save_dir + peft_pubmed + "_seq2seq", peft_pubmed)
ft_model.merge_adapter(peft_pubmed)


print("\n\n\n*********** SUMMARY AFTER TRAINING ***********\n\n\n")
generate_summary(model=ft_model, tokenizer=tokenizer, text=sample.article, truth=sample.abstract)

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
