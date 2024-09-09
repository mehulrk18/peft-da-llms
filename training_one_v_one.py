import os

import adapters
import pandas as pd
import torch
import wandb
from adapters import AutoAdapterModel, AdapterTrainer
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM, \
    DataCollatorForSeq2Seq

from dataset_lib import SumDataLoader, inference_prompt
from peft_module.ahub_pefts import pefts_configuration, PEFTEnum
from testing_scripts.evaluation_metrics_llms import rouge_metric


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


def get_adapter_model(model_id: str, fine_tuning=True, quantization_config=None):
    ddtype = torch.bfloat16  # bfloat16
    compute_dtype = torch.bfloat16  # torch.bfloat16 if bf16 else torch.float32

    if fine_tuning:
        print("Loading Model from Adapter Hub")
        model = AutoAdapterModel.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=ddtype
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=ddtype
        )

    model.config.use_cache = False
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = compute_dtype
    # (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float16))

    return model


def generate_summary(model, tokenizer, content, device):
    # content = f"Summarize the following text:\n\n{text}"
    content = inference_prompt(content)

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
    return summary

    # print("Truth:\n{}\n\n\nPrediction:\n{} ".format(truth, summary))
    #
    # print("\n\n\nRouge Scores: ", rouge.compute(references=[truth], predictions=[summary]))
    # return summary


def llama_model_training(main_directory, training_arguments, fine_tuning=True):
    llama = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: fetch from Configs
    # bits = 4  # 8
    max_seq_len = 512
    dataset_name = "scientific"

    llama_model = get_adapter_model(model_id=llama, fine_tuning=fine_tuning, quantization_config=None)

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(
        llama,
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
        inputs = llama_tokenizer(input_data["text"], max_length=max_seq_len, truncation=True, padding="max_length",
                                 return_tensors="pt")
        return {"input_ids": inputs["input_ids"]}  # , "labels": labels}

    peft_name = None

    # Loading dataset
    data = SumDataLoader(dataset_name=dataset_name, training_samples=2000)
    data.print_dataset_stats()

    # data.train_set, data.validation_set, data.test_set = data.loading_dataset_splits()  # loading data.train_set, data.validation_set, data.test_set
    data.loading_dataset_splits()  # loading data.train_set, data.validation_set, data.test_set

    # Get Tokenized Train and Validation Set
    data.train_set = data.processing_data_with_training_prompt(data.train_set)
    data.validation_set = data.processing_data_with_training_prompt(data.validation_set)
    data.print_dataset_stats()


    # TODO: Add code block for generating Summary with Zero Shot Learning.

    if fine_tuning:
        # Add PEFT from AdapterHub
        lora = {
            "selfattn_lora": True,
            "intermediate_lora": True,
            "output_lora": True,
            "alpha": 16,
            "r": 64,
            "dropout": 0.1,
            "attn_matrices": ["q_proj", "k_proj", "v_proj"]
        }

        adapters.init(model=llama_model)

        peft_config = pefts_configuration[PEFTEnum("lora").name](**lora)

        peft_name = "{}_{}_5".format("scientific", "lora")

        llama_model.add_adapter(peft_name, config=peft_config)
        llama_model.add_causal_lm_head(peft_name)

        llama_model.set_active_adapters(peft_name)
        llama_model.train_adapter(peft_name)
        llama_model.adapter_to(peft_name, device=device)
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



    # Enabling Gradient and


    # TODO: Check Hyperparameters for better results.
    if fine_tuning:
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

        llama_model = trainer.model
        llama_model.merge_adapter(peft_name)
        llama_model.save_adapter(peft_name+"_summarization_2000_5", peft_name)

        train_loss = trainer_stats.training_loss
        print(f"Model Trained with Training loss: {train_loss}")

    # testing the model with Test data.
    # def inference_prompt_processing(sample):
    #     if "sources" in sample.keys():
    #         sample["article"] = sample.pop("sources")
    #
    #     text = [inference_prompt(article=article) for article in sample["article"]]
    #     return {
    #         "text": text
    #     }

    random_ip = """
        Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
    """.strip()

    summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_ip, device=device)
    print("Summmmmmarrryyy: \n", summ)
    try:
        with open("random_ip_{}_{}_{}.txt".format(data.dataset_name, peft_name, ft), "w") as f:
            f.write(summ)
            print("Written Random article summary")
    except:
        pass

    # data.test_set = data.test_set.map(inference_prompt_processing, batch=True)
    df_test_data = pd.DataFrame(data=data.test_set)

    # TODO: write the testing funciton with a metric.
    test_summaries = {
        "truth": [],
        "prediction": []
    }

    # for arxiv and pubmed

    for i in range(len(df_test_data)):
        summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=df_test_data["article"][i], device=device)
        test_summaries["truth"].append(df_test_data["abstract"][i])
        test_summaries["prediction"].append(summary)

    metric = rouge_metric()
    scores = metric.compute(predictions=test_summaries["prediction"], references=test_summaries["truth"])
    df_sum = pd.DataFrame(test_summaries)
    # print("Rouge Scores: ", scores)
    file_name = "Test_summaries_{}_{}.csv".format(data.dataset_name, peft_name if peft_name else "no_finetuning")
    df_sum.to_csv(file_name, index=False)

    print("\n\n\nSummaries with Rouge Score {} saved to file {}!!!!".format(scores, file_name))

    return llama_model


if __name__ == "__main__":
    from datetime import datetime

    # TODO: Add args parser
    try:
        from google.colab import drive

        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        print("Exception: ", e)
        main_directory = ""

    load_dotenv(".env")

    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    login(token=hf_token)
    wandb.login()

    now = datetime.now().strftime("%d-%m-%Y__%H%M%S")
    ft = True # False
    bf16 = False
    bf32 = False
    fp16 = True

    training_args = TrainingArguments(  # Seq2Seq
        remove_unused_columns=False,
        output_dir=main_directory+"results/llama_5_{}_{}_{}".format("arxiv", "lora", now),  # pubmed_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=250,
        learning_rate=1e-4,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=0.3,
        num_train_epochs=5,  # 7
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
        run_name="llama_5_{}_{}_{}_{}".format("arxiv", "lora", "fine_tuned" if ft else "no_fine_tuning", now)
        # push_to_hub=True,
    )

    trained_llama_model = llama_model_training(main_directory=main_directory, training_arguments=training_args,
                                               fine_tuning=ft)

