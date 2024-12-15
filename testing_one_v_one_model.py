import argparse
import logging
import os

import adapters
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from dataset_lib import SumDataLoader, DEFAULT_DOMAIN_PROMPT, DEFAULT_SYSTEM_PROMPT
from utils import generate_summary, rouge_metric, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, torch_dtypes_dict, WandBLogger, check_and_return_df, bertscore_metric, \
    bleu_metric


def testing_model(llama_model, llama_tokenizer, data, peft_full_name, device, logger, chat_template, col_name, metric_name,
                  test_summaries_file_name=None):
    # testing the model with Test data.
    # def inference_prompt_processing(sample):
    #     # if "sources" in sample.keys():
    #     #     sample["article"] = sample.pop("sources")
    #
    #     if chat_template:
    #         from dataset_lib import chat_template_prompt_inference
    #         text = [chat_template_prompt_inference(content=article, system_prompt=DEFAULT_DOMAIN_PROMPT[domain])
    #                 for article in sample["content"]]
    #
    #         return {
    #             "text": text
    #         }
    #     else:
    #         # text = [inference_prompt(article=article) for article in sample["article"]]
    #         from dataset_lib import llama3_testing_prompt
    #         text = [llama3_testing_prompt(content=article, system_prompt=DEFAULT_DOMAIN_PROMPT[domain])
    #                 for article in sample["content"]]
    #
    #         return {
    #             "text": text
    #         }

    if metric_name == "rouge":
        metric = rouge_metric()
    elif metric_name == "bertscore":
        metric = bertscore_metric()
    elif metric_name == "bleu":
        metric = bleu_metric()
    else:
        raise ValueError("Invalid Metric")

    min_samples = data.test_set.num_rows
    if test_summaries_file_name is None:
        test_summaries_file_name = "summaries/summaries_{}_{}_{}samples.csv".format(data.domain.name.lower(),
                                                                                    data.dataset_name.lower(),
                                                                                    min_samples)

    random_text = """
            Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
        """.strip()

    summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_text, device=device,
                            chat_template=chat_template, prompt=DEFAULT_SYSTEM_PROMPT)
    # TODO: write the testing function with a metric.
    test_summaries = {
        "article": [],
        "truth": [],
        col_name: []
    }
    df_sum, file_exists = check_and_return_df(test_summaries_file_name)
    # col_name = col_name + "_shortprompt"
    if col_name not in df_sum.columns:
        try:
            # logger.info("Summary of Random Text from Wikipedia: \n{}".format(summ))
            with open("summaries/random_text_{}.txt".format(peft_full_name), "w") as f:
                f.write("Wikipedia Article: \n{} \n\n\n\n Summary:{}\n".format(random_text, summ))
                logger.info("Written Random article summary")
        except Exception as e:
            logger.error("Exception: ".format(e))
            pass

        # data.test_set = data.test_set.map(inference_prompt_processing, batched=True)
        # df_test_data = pd.DataFrame(data=data.test_set)

        logger.info("PROMPT in USE for Testing: \n'{}'".format(DEFAULT_DOMAIN_PROMPT[data.domain.name]))
        if not file_exists:
            # for i in range(min_samples):
            for i, _obj in enumerate(data.test_set):
                logger.info("Summary for {} sample".format(i+1))
                summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=_obj["content"], device=device,
                                           chat_template=chat_template, prompt=DEFAULT_DOMAIN_PROMPT[data.domain.name])
                test_summaries["article"].append(_obj["content"])
                test_summaries["truth"].append(_obj["summary"])
                test_summaries[col_name].append(summary)
                del summary

        else:
            articles = df_sum["article"]
            i = 0
            for i, art in enumerate(articles):
                logger.info("Summary for {} sample".format(i+1))
                summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=art,
                                           device=device,
                                           chat_template=chat_template, prompt=DEFAULT_DOMAIN_PROMPT[data.domain.name])
                test_summaries[col_name].append(summary)
                del summary

    else:
        logger.info("Summaries in col: {} already exists in file: {}".format(col_name, test_summaries_file_name))
        test_summaries[col_name] = df_sum[col_name]

    scores = 0

    if file_exists:
        test_summaries["truth"] = df_sum["truth"]
    if "mslr" not in peft_full_name:
        # metric = rouge_metric()
        scores = metric.compute(predictions=test_summaries[col_name], references=test_summaries["truth"])
        logger.info("{} Scores: {}".format(metric_name, scores))
    else:
        logger.info("!!! The dataset is MSLR where no reference summaries are available, hence SKIPPING SCORING !!!")

    if file_exists:
        test_summaries.pop("article")
        test_summaries.pop("truth")
        df_sum[col_name] = test_summaries[col_name]

    else:
        df_sum = pd.DataFrame(test_summaries)

        # if "zero_shot" not in peft_full_name:
        #     df_sum = df_sum.remove_columns(["content", "truth"])
    # file_name = "summaries/summaries_{}_{}samples.csv".format(peft_full_name, min_samples)
    df_sum.to_csv(test_summaries_file_name, index=False)

    with open("summaries/{}_scores.txt".format(metric_name), "a") as fp:
        from datetime import datetime
        fp.write("[{}] Summaries of {} for {} samples has {} Scores \n {} \n\n".format(datetime.today().date(),
                                                                                          peft_full_name, min_samples,
                                                                                          metric_name, scores))

    logger.info("\n\n\nSummaries with {} Score {} saved to file {}!!!!".format(metric_name, scores,
                                                                               test_summaries_file_name))


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--checkpoint",type=str, default=None, help="Path of the PT Model Checkpoint to be loaded." )
    parser.add_argument("--trained_peft_path", type=str, help="Path of the PEFT to be loaded.")
    parser.add_argument("--training_samples", type=int, default=1, help="Number of training Samples")
    parser.add_argument("--eval_samples", type=int, default=1, help="Number of Evaluation Samples")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of Samples to be tested")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Torch Data Type to be used")
    parser.add_argument("--metric", type=str, required=True, choices=["rouge", "bertscore", "bleu"],
                        help="Torch Data Type to be used")
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize the model")
    parser.add_argument("--sorted_dataset", type=bool, default=False, help="do you want to sort the dataset?")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    parser.add_argument("--mlm", type=bool, default=False, help="Using attention mask")

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        main_directory = ""

    args = parser.parse_args()
    trained_peft_path = main_directory + args.trained_peft_path
    mlm = True if "mlm" in trained_peft_path or args.mlm else False
    model_checkpoint = args.checkpoint
    training_samples = args.training_samples
    eval_samples = args.eval_samples
    test_samples = args.test_samples
    sort_data = args.sorted_dataset
    quantize = args.quantize
    metric = args.metric
    torch_dtype = torch_dtypes_dict[args.torch_dtype]
    chat_template = True if "chat_template" in trained_peft_path or args.chat_template else False
    use_instruct_model = True if "instruct" in trained_peft_path or args.chat_template else False
    # provider = "hf" if "hf" in trained_peft_path else "ah"

    peft_path_splits = trained_peft_path.split("/")
    # if peft_path_splits[0] == "results":
    #     peft_dir = "_".join(peft_path_splits)
    #     domain = peft_path_splits[3].split("_")[0]
    #     adapter_name = peft_path_splits[3]

    # elif trained_peft_path.split("/")[0] ==  :# "saved_models":
    peft_dir = peft_path_splits[1]
    provider, domain, dataset_name, peft_type = tuple(peft_dir.split("_")[:4])
    adapter_name = "{}_{}_{}".format(domain, dataset_name, peft_type)

    load_dotenv(".env")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    run_name = 'testing_{}_{}samples.log'.format("_".join(peft_path_splits), test_samples)
    wnb_run = wandb.init(name=run_name)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    logging.basicConfig(
        filename=main_directory + 'logs/{}'.format(run_name),  # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    logger = logging.getLogger()
    wnb = WandBLogger()
    wnb.wandb = wandb

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the log level for the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(wnb)

    logger.info("Device in use: {}".format(device))
    # llama_model = get_pretrained_model(ah=ah)
    llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize,
                            model_checkpoint=model_checkpoint, mlm=mlm, torch_dtype=torch_dtype)
    # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

    logger.info("Check point MODEL: \n{}".format(llama.model))

    if provider == "hf":
        # Method 1 - HuggingFace
        from peft import PeftModel
        # llama.model = PeftModel.from_pretrained(llama.model, trained_peft_path, adapter_name=adapter_name) #, use_safetensors=True)
        # llama.model = llama.model.merge_and_unload()
        llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
        llama.model.set_adapter([adapter_name])
        llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name=adapter_name,
                                                                  torch_dtype=torch_dtype)
        llama.model = llama.model.to(torch_dtype)
    else:
        # Method 2 - AdapterHub
        adapters.init(model=llama.model)
        loaded_peft = llama.model.load_adapter(trained_peft_path, with_head=False)
        llama.model.set_active_adapters([loaded_peft])
        llama.model.adapter_to(loaded_peft, device=device)

        llama.model = llama.model.to(torch.bfloat16)

        llama.model.enable_input_require_grads()
        llama.model.gradient_checkpointing_enable()
        logger.info("\nMethod 2 LLaMA Model's Summary:\n{}\n\n\n".format(llama.model.adapter_summary()))

    logger.info("Loaded MODEL: \n{}".format(llama.model))

    if chat_template:
        logger.info("****** RESULTS ARE GENERATED USING CHAT TEMPLATE ******")

    data = SumDataLoader(domain=domain, dataset_name=dataset_name, training_samples=training_samples,
                         eval_samples=eval_samples, test_samples=test_samples, sort_dataset_on_article_len=sort_data,
                         chat_template=chat_template)
    # data.loading_dataset_splits()

    data.train_set = None
    data.validation_set = None

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data=data, peft_full_name=peft_dir,
                  col_name=peft_type, logger=logger, device=device, chat_template=chat_template, metric_name=metric,
                  test_summaries_file_name=None)
    wnb_run.finish()
