import os

import adapters
import pandas as pd
import torch
import wandb
from adapters import AutoAdapterModel, AdapterTrainer
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dataset_lib import SumDataLoader
from peft_module.ahub_pefts import pefts_configuration, PEFTEnum


# Model Config
# llama31 = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # works only with transformers==4.43.3
# llama3 = "meta-llama/Meta-Llama-3-8B-Instruct"
# llama2 = "meta-llama/Llama-2-7b-hf"
# mistral = "mistralai/Mistral-7B-v0.3"


# model_name_dict = {
#     llama3: "LLaMA3",
#     llama31: "LLaMA3.1",
#     llama2: "LLaMA2",
#     mistral: "Mistral"
# }


def get_adapter_model(model_id: str):
    ddtype = torch.bfloat16  # bfloat16
    compute_dtype = torch.bfloat16  # torch.bfloat16 if bf16 else torch.float32

    model = AutoAdapterModel.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=None,
        torch_dtype=ddtype
    )

    model.config.use_cache = False
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = compute_dtype
    # (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float16))

    return model


def llama_model_training(main_directory, training_arguments):
    load_dotenv(".env")

    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    login(token=hf_token)
    wandb.login()

    llama = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: fetch from Configs
    # bits = 4  # 8
    max_seq_len = 1024
    dataset_name = "arxiv"
    bf16 = False
    bf32 = False
    fp16 = True

    llama_model = get_adapter_model(model_id=llama)

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(
        llama,
        padding_side="right",
        tokenizer_type="llama",
        trust_remote_code=True,
        use_fast=True
    )

    # Loading dataset
    data = SumDataLoader(dataset_name="scientific", training_samples=5000, force_download=True)

    data = data.loading_dataset_splits()  # loading data.train_set, data.validation_set, data.test_set

    # Preprocessing the data splits for training and validation with prompts
    preprocessing_function = data.get_preprocess_function()

    # Get Tokenized Train and Validation Set
    data.train_set = data.processing_data_with_prompt(data.train_set)
    data.validation_set = data.processing_data_with_prompt(data.validation_set)

    df_test_data = pd.DataFrame(data=data.test_set)

    # TODO: Add code block for generating Summary with Zero Shot Learning.

    # Add PEFT from AdapterHub
    lora = {
        "selfattn_lora": True,
        "intermediate_lora": True,
        "output_lora": True,
        "alpha": 16,
        "r": 32,
        "dropout": 0.1,
        "attention_matrices": ["q", "k", "v"]
    }

    adapters.init(model=llama_model)

    peft_config = pefts_configuration[PEFTEnum("lora").name](**lora)

    peft_name = "{}_{}".format("scientific", "lora")

    llama_model.add_adapter(peft_name, config=peft_config)
    llama_model.add_causal_lm_head(peft_name)

    llama_model.set_active_adapters([peft_name])
    llama_model.train_adapter([peft_name])
    llama_model.adapter_to(peft_name, device=device)

    llama_model.gradient_checkpointing_enable()
    llama_model.enable_input_require_grads()

    for name, param in llama_model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameters of the model: {name} - {param.shape}")
    print("\nLLaMA Model's Summary:\n", llama_model.adapter_summary())


    # Enabling Gradient and
    torch.autograd.set_detect_anomaly(True)
    torch.enable_grad()


    # TODO: Check Hyperparameters for better results.

    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_seq_len,
    #                                        label_pad_token_id=-100)
    data_collator = DataCollatorForLanguageModeling(llama_tokenizer, mlm=False)
    trainer = AdapterTrainer(  # AdapterTrainer # Seq2SeqTrainer(
        model=llama_model,
        tokenizer=llama_tokenizer,
        data_collator=data_collator,
        train_dataset=data.train_set,
        eval_dataset=data.test_set,
        args=training_arguments
    )

    # accelerator.prepare(trainer)
    trainer_stats = trainer.train()

    results = trainer.evaluate()
    print("Results from Training: \n", results)

    train_loss = trainer_stats.training_loss
    llama_model = trainer.model
    llama_model.merge_adapter(peft_name)
    llama_model.save_adapter(peft_name+"_summarization", peft_name)

    print(f"Model Trained with Training loss: {train_loss}")

    return llama_model


if __name__ == "__main__":

    # TODO: Add args parser
    try:
        from google.colab import drive

        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        print("Exception: ", e)
        main_directory = ""

    training_args = TrainingArguments(  # Seq2Seq
        remove_unused_columns=False,
        output_dir=main_directory+"results/llama_{}_{}".format("arxiv", "lora"),  # pubmed_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=7,  # 7
        evaluation_strategy="epoch",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        logging_dir=main_directory+"logs/",
        # output_dir="results/",
        report_to="wandb",
        save_safetensors=True,
        lr_scheduler_type="constant",  # "cosine",
        seed=42,
        load_best_model_at_end=True,
        # push_to_hub=True,
    )

    trained_llama_model = llama_model_training(main_directory=main_directory, training_arguments=training_args)

