import argparse
import logging
import os
import statistics

import adapters
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from dataset_lib import SumDataLoader, DEFAULT_DOMAIN_PROMPT, DEFAULT_SYSTEM_PROMPT, SumDomains, datasets_info_dict
from utils import generate_summary, rouge_metric, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, torch_dtypes_dict, WandBLogger, check_and_return_df, bertscore_metric, \
    bleu_metric, bleurt_metric, read_yaml


def unseen_test_data_inference(llama_model, llama_tokenizer, data_class, peft_full_name, device, logger, chat_template,
                               col_name, metric_name):
    if metric_name == "rouge":
        metric = rouge_metric()
    elif metric_name == "bertscore":
        metric = bertscore_metric()
    elif metric_name == "bleu":
        metric = bleu_metric()
    # elif metric_name == "bleurt":
    #     metric = bleurt_metric()
    elif metric_name == "all":
        rouge = rouge_metric()
        bertscore = bertscore_metric()
        bleu = bleu_metric()
        # bleurt = bleurt_metric()
    else:
        raise ValueError("Invalid Metric")

    save_df, file_exists = True, False
    test_summaries_file_name = "summaries/summaries_{}_{}.csv".format(data_class.domain.lower(), data_class.name.lower())
    data = pd.read_csv(data_class.local_path)
    min_samples = data.test_set.num_rows



    # if test_summaries_file_name is None:
    #     test_summaries_file_name = "summaries/summaries_{}_{}_{}samples.csv".format(data.domain.name.lower(),
    #                                                                                 data.dataset_name.lower(),
    #                                                                                 min_samples)

    # random_text = """
    #         Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
    #     """.strip()
    #
    # summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_text, device=device,
    #                         chat_template=chat_template, prompt=DEFAULT_SYSTEM_PROMPT)
    # TODO: write the testing function with a metric.
    test_summaries = {
        # "article": [],
        # "truth": [],
        col_name: []
    }
    df_sum, file_exists = check_and_return_df(test_summaries_file_name)
    # col_name = col_name + "_shortprompt"
    if col_name not in df_sum.columns:
        logger.info("PROMPT in USE for Testing: \n'{}'".format(DEFAULT_DOMAIN_PROMPT[data_class.name.upper()]))
        articles = data["content"]
        i = 0
        for i, art in enumerate(articles):
            logger.info("Summary for {} sample".format(i+1))
            summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=art,
                                       device=device,
                                       chat_template=chat_template, prompt=DEFAULT_DOMAIN_PROMPT[data_class.name.upper()])
            test_summaries[col_name].append(summary)
            del summary
    else:
        logger.info("Summaries in col: {} already exists in file: {}".format(col_name, test_summaries_file_name))
        test_summaries[col_name] = data[col_name]
        save_df = False

    scores, rouge_scores, bertscore_scores, bleu_scores, bleurt_scores = 0, 0, 0, 0, 0
    truth = data["summary"]
        # metric = rouge_metric()
    if metric_name == "all":
        rouge_scores = rouge.compute(predictions=test_summaries[col_name], references=truth)
        bertscore_scores = bertscore.compute(predictions=test_summaries[col_name],
                                             references=truth, lang="en", verbose=True)

        bertscore_scores["precision"] = {"mean": statistics.mean(bertscore_scores["precision"]),
                                         "median": statistics.median(bertscore_scores["precision"])}
        bertscore_scores["recall"] = {"mean": statistics.mean(bertscore_scores["recall"]),
                                      "median": statistics.median(bertscore_scores["recall"])}
        bertscore_scores["f1"] = {"mean": statistics.mean(bertscore_scores["f1"]),
                                  "median": statistics.median(bertscore_scores["f1"])}

        bleu_scores = bleu.compute(predictions=test_summaries[col_name], references=truth)
        # bleurt_scores = bleurt.compute(predictions=test_summaries[col_name], references=truth)
        # bleurt_scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
        #                     "median": statistics.median(bleurt_scores["scores"])}

        logger.info("ROUGE Scores: {}".format(rouge_scores))
        logger.info("BERTSCORE Scores: {}".format(bertscore_scores))
        logger.info("BLEU Scores: {}".format(bleu_scores))
        # logger.info("BLEURT Scores: {}".format(bleurt_scores))
    else:
        if metric_name == "bertscore":
            bertscore_scores = metric.compute(predictions=test_summaries[col_name], references=truth,
                                    lang="en", verbose=True)
            scores = {}
            scores["precision"] = {"mean": statistics.mean(bertscore_scores["precision"]),
                                   "median": statistics.median(bertscore_scores["precision"])}
            scores["recall"] = {"mean": statistics.mean(bertscore_scores["recall"]),
                                "median": statistics.median(bertscore_scores["recall"])}
            scores["f1"] = {"mean": statistics.mean(bertscore_scores["f1"]),
                            "median": statistics.median(bertscore_scores["f1"])}
        # elif metric_name == "bleurt":
        #     bleurt_scores = metric.compute(predictions=test_summaries[col_name], references=truth)
        #     scores = {}
        #     scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
        #                         "median": statistics.median(bleurt_scores["scores"])}
        else:
            scores = metric.compute(predictions=test_summaries[col_name], references=truth)
        logger.info("{} Scores: {}".format(metric_name, scores))

    if file_exists:
        test_summaries.pop("content")
        test_summaries.pop("summary")
    else:
        df_sum["article"] = data["content"]
        df_sum["truth"] = truth
    df_sum[col_name] = test_summaries[col_name]

    if save_df:
        df_sum.to_csv(test_summaries_file_name, index=False)

    # TODO: Write Scores to a CSV file directly, without storing it in a txt file.
    if metric_name == "all":
        """
        unseen_metrics_results_{}.csv # scientific, medical, legal, news
        
        peft_name | rouge_1 | rouge_2 | rouge_L | rouge_Lsum | bertscore_precision_mean | bertscore_precision_median | bertscore_recall_mean | bertscore_recall_median | bertscore_f1_mean | bertscore_f1_median | bleu | bleu_ngram_precision | bleurt_mean | bleurt_median
        
        """



        from datetime import datetime
        with open("summaries/rouge_scores.txt", "a") as fp:
            fp.write("[{}] Summaries of {} for {} samples has rouge Scores \n {} \n\n".format(datetime.today().date(),
                                                                                              peft_full_name,
                                                                                              min_samples,
                                                                                              rouge_scores))

        logger.info("\n\n\nSummaries with rouge Score {} saved to file {}!!!!".format(rouge_scores,
                                                                                      test_summaries_file_name))

        with open("summaries/bertscore_scores.txt", "a") as fp:
            fp.write("[{}] Summaries of {} for {} samples has bertscore Scores \n {} \n\n".format(datetime.today().date(),
                                                                                                  peft_full_name,
                                                                                                  min_samples,
                                                                                                  bertscore_scores))

        logger.info("\n\n\nSummaries with bertscore Score {} saved to file {}!!!!".format(bertscore_scores,
                                                                                          test_summaries_file_name))

        with open("summaries/bleu_scores.txt", "a") as fp:
            fp.write("[{}] Summaries of {} for {} samples has bleu Scores \n {} \n\n".format(datetime.today().date(),
                                                                                             peft_full_name, min_samples,
                                                                                             bleu_scores))
        logger.info("\n\n\nSummaries with bleu Score {} saved to file {}!!!!".format(bleu_scores,
                                                                                     test_summaries_file_name))

        # with open("summaries/bleurt_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has bleuRT Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                      peft_full_name, min_samples,
        #                                                                                      bleurt_scores))
        # logger.info("\n\n\nSummaries with bleuRT Score {} saved to file {}!!!!".format(bleurt_scores,
        #                                                                                test_summaries_file_name))

    else:
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

    parser.add_argument("--config_file_path", type=str, help="Path of the config file containing pefts and dataset.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Torch Data Type to be used")
    parser.add_argument("--metric", type=str, required=True, choices=["rouge", "bertscore", "bleu", "bleurt", "all"],
                        help="Metric to be used for testing, pass 'all' if you want test on all")
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize the model")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    parser.add_argument("--mlm", type=bool, default=False, help="Using attention mask")

    main_directory = ""

    args = parser.parse_args()
    config_file = main_directory + args.config_file_path
    mlm = True if "mlm" in args.mlm else False
    sort_data = args.sorted_dataset
    quantize = args.quantize
    metric = args.metric
    torch_dtype = torch_dtypes_dict[args.torch_dtype]
    chat_template = True # if "chat_template" in trained_peft_path or args.chat_template else False
    use_instruct_model = True # if "instruct" in trained_peft_path or args.chat_template else False
    # provider = "hf" if "hf" in trained_peft_path else "ah"
    configs = read_yaml(file_name=config_file)
    dataset_name = configs["dataset_name"]
    zero_shot = configs["zero_shot"]

    # elif trained_peft_path.split("/")[0] ==  :# "saved_models":
    peft_names = []
    if not zero_shot:
        for path in configs["pefts"]:
            a_name = path.split("/")[-1]
            peft_names.append(a_name)

    provider = "hf"

    load_dotenv(".env")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    run_name = 'unseen_data_inference_{}_{}.log'.format("-".join(peft_names) if not zero_shot else "zero_shot", dataset_name)
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
    llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize, mlm=mlm, torch_dtype=torch_dtype)
    # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

    logger.info("Check point MODEL: \n{}".format(llama.model))

    # Method 1 - HuggingFace
    if not zero_shot:
        for a_path, a_name in zip(configs["paths"], peft_names):
            llama.model.load_adapter(a_path, adapter_name=a_name)
    # llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
    # llama.model.set_adapter([adapter_name])
    # llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name=adapter_name,
    #                                                           torch_dtype=torch_dtype)
        llama.model = llama.model.to(torch_dtype)

    logger.info("Loaded MODEL: \n{}".format(llama.model))

    if chat_template:
        logger.info("****** RESULTS ARE GENERATED USING CHAT TEMPLATE ******")

    # data = SumDataLoader(domain=domain, dataset_name=dataset_name, training_samples=training_samples,
    #                      eval_samples=eval_samples, test_samples=test_samples, sort_dataset_on_article_len=sort_data,
    #                      chat_template=chat_template)

    domain = SumDomains("unseen_test")
    data_class = datasets_info_dict[domain][dataset_name]
    # data.loading_dataset_splits()


    # data.train_set = None
    # data.validation_set = None

    unseen_test_data_inference(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data_class=data_class,
                               peft_full_name= "-".join(peft_names) if not zero_shot else "zero_shot",
                               col_name="-".join(peft_names) if not zero_shot else "zero_shot", logger=logger,
                               device=device, chat_template=chat_template, metric_name=metric)
    wnb_run.finish()
