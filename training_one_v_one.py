import argparse
import os

import adapters
import torch
import wandb
from adapters import AdapterTrainer
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dataset_lib import SumDataLoader
from peft_module.ahub_pefts import pefts_configuration, PEFTEnum
from utils import read_yaml, get_pretrained_model, MODEL_ID

PEFT_CONFIGS_FILE = "configs/peft_configs.yaml"
global MAX_SEQ_LENGTH


def llama_model_training(main_directory, training_arguments, training_samples, peft_name, domain, fine_tuning=True, save_peft_name=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: fetch from Configs
    # bits = 4  # 8
    llama_model = get_pretrained_model(fine_tuning=fine_tuning, quantization_config=None)

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        padding_side="right",
        tokenizer_type="llama",
        trust_remote_code=True,
        use_fast=True
    )
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.add_special_tokens({
        "eos_token": llama_tokenizer.convert_ids_to_tokens(llama_model.config.eos_token_id),
        "bos_token": llama_tokenizer.convert_ids_to_tokens(llama_model.config.bos_token_id),
    })

    def tokenization_process(input_data):
        # inputs = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True)

        inputs = llama_tokenizer(input_data["text"], max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length",
                                 return_tensors="pt")
        return {"input_ids": inputs["input_ids"]}  # , "labels": labels}

    # peft_name = None

    # Loading dataset
    data = SumDataLoader(dataset_name=domain, training_samples=training_samples)
    data.print_dataset_stats()

    # data.train_set, data.validation_set, data.test_set = data.loading_dataset_splits()  # loading data.train_set, data.validation_set, data.test_set
    data.loading_dataset_splits()  # loading data.train_set, data.validation_set, data.test_set

    # Get Tokenized Train and Validation Set
    data.train_set = data.processing_data_with_training_prompt(data.train_set)
    data.validation_set = data.processing_data_with_training_prompt(data.validation_set)
    data.print_dataset_stats()

    # TODO: Add code block for generating Summary with Zero Shot Learning.

    if fine_tuning:

        pefts_from_yaml = read_yaml(file_name=PEFT_CONFIGS_FILE)

        # Add PEFT from AdapterHub
        peft_configs = pefts_from_yaml[peft_name]

        adapters.init(model=llama_model)

        config = pefts_configuration[PEFTEnum(peft_name).name](**peft_configs)

        peft_layer_name = "{}_{}".format(domain, peft_name)

        llama_model.add_adapter(peft_layer_name, config=config)
        llama_model.add_causal_lm_head(peft_layer_name)

        llama_model.set_active_adapters(peft_layer_name)
        llama_model.train_adapter(peft_layer_name)
        llama_model.adapter_to(peft_layer_name, device=device)
        print("\nLLaMA Model's Summary:\n", llama_model.adapter_summary())
        llama_model.enable_input_require_grads()
        llama_model.gradient_checkpointing_enable()
        for param in llama_model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        # for name, param in llama_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable parameters of the model: {name} - {param.shape}")

        # TODO: Check Hyperparameters for better results.
        # tokenize dataset
        data.train_set, data.validation_set, data.test_set = data.tokenization_of_data_splits(tokenization_process=tokenization_process)
        data.print_dataset_stats()
        # torch.autograd.set_detect_anomaly(True)
        torch.enable_grad()
        # data_collator = DataCollatorForSeq2Seq(llama_tokenizer, model=llama_model, padding="max_length", max_length=max_seq_len,
        #                                        label_pad_token_id=-100)
        data_collator = DataCollatorForLanguageModeling(llama_tokenizer, mlm=False, return_tensors="pt")
        trainer = AdapterTrainer(  # AdapterTrainer # Seq2SeqTrainer(
            model=llama_model,
            tokenizer=llama_tokenizer,
            data_collator=data_collator,
            train_dataset=data.train_set,
            eval_dataset=data.validation_set,
            args=training_arguments
        )

        from accelerate import Accelerator
        accelerator = Accelerator()
        accelerator.prepare(trainer)

        trainer_stats = trainer.train()

        results = trainer.evaluate()
        print("Results from Training: \n", results)

        if save_peft_name is None:
            save_peft_name = peft_layer_name + "_temp_summarization"
        llama_model = trainer.model
        llama_model.merge_adapter(peft_layer_name)
        llama_model.save_adapter(main_directory+"saved_models/"+save_peft_name, peft_layer_name)

        train_loss = trainer_stats.training_loss
        print(f"Model Trained with Training loss: {train_loss}")

    return llama_model


if __name__ == "__main__":
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    global MAX_SEQ_LENGTH

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--peft", type=str, default=None, help="peft name for config_file")
    parser.add_argument("--domain", type=str, default=None, help="Domain name for dataset")
    parser.add_argument("--train_epochs", type=int, default=1, help="Training Epochs")
    parser.add_argument("--ft", type=bool, default=True, help="Finetune the model or not")
    parser.add_argument("--training_samples", type=int, default=1000, help="Number of training Samples")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Context window length")


    # TODO: Add args parser
    try:
        from google.colab import drive

        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        print("Exception: ", e)
        main_directory = ""

    args = parser.parse_args()

    print("Args: \n", args)


    peft_name = args.peft
    domain = args.domain
    training_epochs = args.train_epochs
    ft = args.ft  # False
    training_samples = args.training_samples
    MAX_SEQ_LENGTH = args.max_seq_len

    save_peft_name = "{}_{}_{}_{}_summarization".format(domain, peft_name, training_samples, training_epochs)

    if peft_name is None:
        raise Exception("PEFT NAME NOT FOUND!! Provide one, your options are [lora, ia3, simple_adapter, reft]")

    if domain is None:
        raise Exception("DOMAIN NAME NOT FOUND!! Provide one, your options are [scientific, medical, legal, news]")

    from datetime import datetime

    load_dotenv(".env")

    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    login(token=hf_token)
    wandb.login()

    now = datetime.now().strftime("%d-%m-%Y")

    bf16 = False
    bf32 = False
    fp16 = True

    training_args = TrainingArguments(  # Seq2Seq
        remove_unused_columns=False,
        output_dir=main_directory+"results/llama_{}_{}_{}_{}_{}".format(domain, peft_name, training_samples, training_epochs, now),  # pubmed_lora",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        logging_steps=250,
        learning_rate=2e-4,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=0.3,
        num_train_epochs=training_epochs,  # 7
        evaluation_strategy="epoch",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        save_total_limit=1,
        group_by_length=True,
        logging_dir=main_directory+"logs/",
        report_to="wandb",
        save_safetensors=True,
        lr_scheduler_type="constant",  # "cosine",
        seed=42,
        load_best_model_at_end=True,
        run_name="llama_{}_{}_{}_{}_{}_{}".format(domain, peft_name, "fine_tuned" if ft else "no_fine_tuning",
                                                  training_samples if ft else "", training_epochs if ft else "", now)
        # push_to_hub=True,
    )

    trained_llama_model = llama_model_training(main_directory=main_directory, training_arguments=training_args,
                                               training_samples=training_samples, fine_tuning=ft, peft_name=peft_name,
                                               domain=domain, save_peft_name=save_peft_name)

    # print("\n\nTrained LLaMA Model: \n", trained_llama_model.adapter_summary(as_dict=True))
    print("\n\n Trained LLaMA Model: \n", trained_llama_model)

