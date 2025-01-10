import argparse
import logging
import os
import statistics

import adapters
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from dataset_lib import SumDataLoader, DEFAULT_DOMAIN_PROMPT, DEFAULT_SYSTEM_PROMPT
from utils import generate_summary, rouge_metric, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, torch_dtypes_dict, WandBLogger, check_and_return_df, bertscore_metric, \
    bleu_metric, bleurt_metric


def testing_model(llama_model, llama_tokenizer, data, peft_full_name, device, logger, chat_template, col_name, metric_name,
                  test_summaries_file_name=None, overwrite_results=False):

    if metric_name == "rouge":
        metric = rouge_metric()
    elif metric_name == "bertscore":
        metric = bertscore_metric()
    elif metric_name == "bleu":
        metric = bleu_metric()
    elif metric_name == "bleurt":
        metric = bleurt_metric()
    elif metric_name == "all":
        rouge = rouge_metric()
        bertscore = bertscore_metric()
        bleu = bleu_metric()
        # bleurt = bleurt_metric()
    else:
        raise ValueError("Invalid Metric")

    save_df = True
    min_samples = data.test_set.num_rows
    if test_summaries_file_name is None:
        test_summaries_file_name = "summaries/summaries_{}_{}_{}samples.csv".format(data.domain.name.lower(),
                                                                                    data.dataset_name.lower(),
                                                                                    min_samples)

    # random_text = """
    #         Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
    #     """.strip()
    #
    # summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_text, device=device,
    #                         chat_template=chat_template, prompt=DEFAULT_SYSTEM_PROMPT)
    # TODO: write the testing function with a metric.
    test_summaries = {
        "article": [],
        "truth": [],
        col_name: []
    }
    df_sum, file_exists = check_and_return_df(test_summaries_file_name)
    # col_name = col_name + "_shortprompt"
    if col_name not in df_sum.columns or overwrite_results:
        # try:
        #     # logger.info("Summary of Random Text from Wikipedia: \n{}".format(summ))
        #     with open("summaries/random_text_{}.txt".format(peft_full_name), "w") as f:
        #         f.write("Wikipedia Article: \n{} \n\n\n\n Summary:{}\n".format(random_text, summ))
        #         logger.info("Written Random article summary")
        # except Exception as e:
        #     logger.error("Exception: ".format(e))
        #     pass

        # data.test_set = data.test_set.map(inference_prompt_processing, batched=True)
        # df_test_data = pd.DataFrame(data=data.test_set)

        logger.info("PROMPT in USE for Testing: \n'{}'".format(DEFAULT_DOMAIN_PROMPT[data.domain.name]))
        if not file_exists:
            logger.info("File DOES NOT EXIST: {}, hence generating "
                        "summaries for {} samples".format(test_summaries_file_name, min_samples))
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
            logger.info(
                "File EXISTS: {}, generating summaries for {} samples".format(test_summaries_file_name,
                                                                                             min_samples))
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
        save_df = False

    scores, rouge_scores, bertscore_scores, bleu_scores, bleurt_scores = 0, 0, 0, 0, 0
    if file_exists:
        test_summaries["truth"] = df_sum["truth"]
    # if "mslr" not in peft_full_name:
        # metric = rouge_metric()
    if metric_name == "all":
        rouge_scores = rouge.compute(predictions=test_summaries[col_name], references=test_summaries["truth"])
        bertscore_scores = bertscore.compute(predictions=test_summaries[col_name],
                                             references=test_summaries["truth"], lang="en", verbose=True)

        bertscore_scores["precision"] = {"mean": statistics.mean(bertscore_scores["precision"]),
                                         "median": statistics.median(bertscore_scores["precision"])}
        bertscore_scores["recall"] = {"mean": statistics.mean(bertscore_scores["recall"]),
                                      "median": statistics.median(bertscore_scores["recall"])}
        bertscore_scores["f1"] = {"mean": statistics.mean(bertscore_scores["f1"]),
                                  "median": statistics.median(bertscore_scores["f1"])}

        bleu_scores = bleu.compute(predictions=test_summaries[col_name], references=test_summaries["truth"])
        # bleurt_scores = bleurt.compute(predictions=test_summaries[col_name], references=test_summaries["truth"])
        # bleurt_scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
        #                     "median": statistics.median(bleurt_scores["scores"])}

        logger.info("ROUGE Scores: {}".format(rouge_scores))
        logger.info("BERTSCORE Scores: {}".format(bertscore_scores))
        logger.info("BLEU Scores: {}".format(bleu_scores))
        # logger.info("BLEURT Scores: {}".format(bleurt_scores))
    else:
        if metric_name == "bertscore":
            bertscore_scores = metric.compute(predictions=test_summaries[col_name], references=test_summaries["truth"],
                                    lang="en", verbose=True)
            scores = {}
            scores["precision"] = {"mean": statistics.mean(bertscore_scores["precision"]),
                                   "median": statistics.median(bertscore_scores["precision"])}
            scores["recall"]  = {"mean": statistics.mean(bertscore_scores["recall"]),
                                 "median": statistics.median(bertscore_scores["recall"])}
            scores["f1"] =  {"mean": statistics.mean(bertscore_scores["f1"]),
                             "median": statistics.median(bertscore_scores["f1"])}
        elif metric_name == "bleurt":
            raise ValueError("BLEURT is not supported in the current version")
        #     bleurt_scores = metric.compute(predictions=test_summaries[col_name], references=test_summaries["truth"])
        #     scores = {}
        #     scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
        #                            "median": statistics.median(bleurt_scores["scores"])}
        else:
            scores = metric.compute(predictions=test_summaries[col_name], references=test_summaries["truth"])
        logger.info("{} Scores: {}".format(metric_name, scores))
    # else:
    #     logger.info("!!! The dataset is MSLR where no reference summaries are available, hence SKIPPING SCORING !!!")

    if file_exists:
        test_summaries.pop("article")
        test_summaries.pop("truth")
        df_sum[col_name] = test_summaries[col_name]

    else:
        df_sum = pd.DataFrame(test_summaries)

    # TODO: understand where is the mistake and fix it.
    if save_df or overwrite_results:
        df_sum.to_csv(test_summaries_file_name, index=False)
        # if "zero_shot" not in peft_full_name:
        #     df_sum = df_sum.remove_columns(["content", "truth"])
    # file_name = "summaries/summaries_{}_{}samples.csv".format(peft_full_name, min_samples)

    if metric_name == "all":
        from datetime import datetime
        # with open("summaries/rouge_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has rouge Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                       peft_full_name,
        #                                                                                       min_samples,
        #                                                                                       rouge_scores))

        # ROUGE
        logger.info("Writing ROUGE Scores {} to file: summaries/rouge_scores.csv".format(rouge_scores))
        rouge_df = pd.read_csv("summaries/rouge_scores.csv")
        new_row = {
            "model": peft_full_name,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "rougeLsum": rouge_scores["rougeLsum"]
        }
        if peft_full_name in rouge_df["model"].values:
            # Update the existing row
            rouge_df.loc[rouge_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
        else:
            # Add a new row
            rouge_df = pd.concat([rouge_df, pd.DataFrame([new_row])], ignore_index=True)
        rouge_df.to_csv("summaries/rouge_scores.csv", index=False)

        logger.info("\n\n\nSummaries with rouge Score {} saved to file {}!!!!".format(rouge_scores,
                                                                                      test_summaries_file_name))

        # with open("summaries/bertscore_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has bertscore Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                           peft_full_name,
        #                                                                                           min_samples,
        #                                                                                           bertscore_scores))
        # BERTSCORE
        bertscore_df = pd.read_csv("summaries/bertscore_scores.csv")
        logger.info("Writing BERTScores {} to file: summaries/bertscore_scores.csv".format(bertscore_scores))
        new_row = {
            "model": peft_full_name,
            "precision_mean": bertscore_scores["precision"]["mean"],
            "precision_median": bertscore_scores["precision"]["median"],
            "recall_mean": bertscore_scores["recall"]["mean"],
            "recall_median": bertscore_scores["recall"]["median"],
            "f1_mean": bertscore_scores["f1"]["mean"],
            "f1_median": bertscore_scores["f1"]["median"]
        }
        if peft_full_name in bertscore_df["model"].values:
            # Update the existing row
            bertscore_df.loc[bertscore_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
        else:
            # Add a new row
            bertscore_df = pd.concat([bertscore_df, pd.DataFrame([new_row])], ignore_index=True)
        bertscore_df.to_csv("summaries/bertscore_scores.csv", index=False)

        # TODO: Add the scores to the bertscore_scores.csv file
        logger.info("\n\n\nSummaries with bertscore Score {} saved to file {}!!!!".format(bertscore_scores,
                                                                                          test_summaries_file_name))

        # BLEU
        # with open("summaries/bleu_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has bleu Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                      peft_full_name, min_samples,
        #                                                                                      bleu_scores))
        logger.info("Writing Bleu Scores {} to file: summaries/bleu_scores.csv".format(bleu_scores))
        bleu_df = pd.read_csv("summaries/bleu_scores.csv")
        new_row = {
            "model": peft_full_name,
            "bleu": bleu_scores["bleu"],
            "precisions": bleu_scores["precisions"],
            "brevity_penalty": bleu_scores["brevity_penalty"],
            "length_ratio": bleu_scores["length_ratio"],
            "translation_length": bleu_scores["translation_length"],
            "reference_length": bleu_scores["reference_length"]
        }
        if peft_full_name in bleu_df["model"].values:
            # Update the existing row
            bleu_df.loc[bleu_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
        else:
            # Add a new row
            bleu_df = pd.concat([bleu_df, pd.DataFrame([new_row])], ignore_index=True)
        bleu_df.to_csv("summaries/bleu_scores.csv", index=False)
        logger.info("\n\n\nSummaries with bleu Score {} saved to file {}!!!!".format(bleu_scores,
                                                                                     test_summaries_file_name))

        # TODO: Add the scores to the bleu_scores.csv file
        # BLEURT
        # with open("summaries/bleurt_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has bleuRT Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                      peft_full_name, min_samples,
        #                                                                                      bleurt_scores))

        # TODO: Uncomment After fixing
        # logger.info("Writing BleuRT Scores {} to file: summaries/bleurt_scores.csv".format(bleurt_scores))
        # bleurt_df = pd.read_csv("summaries/bleurt_scores.csv")
        # new_row = {
        #     "model": peft_full_name,
        #     "mean": bleurt_scores["scores"]["mean"],
        #     "median": bleurt_scores["scores"]["median"],
        # }
        # if peft_full_name in bleurt_df["model"].values:
        #     # Update the existing row
        #     bleurt_df.loc[bleurt_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
        # else:
        #     # Add a new row
        #     bleurt_df = pd.concat([bleurt_df, pd.DataFrame([new_row])], ignore_index=True)
        # bleurt_df.to_csv("summaries/bleurt_scores.csv", index=False)
        # logger.info("\n\n\nSummaries with bleuRT Score {} saved to file {}!!!!".format(scores,
        #                                                                              test_summaries_file_name))

    else:
        # with open("summaries/{}_scores.txt".format(metric_name), "a") as fp:
        #     from datetime import datetime
        #     fp.write("[{}] Summaries of {} for {} samples has {} Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                    peft_full_name, min_samples,
        #                                                                                    metric_name, scores))
        if metric_name == "rouge":
            df_file = "summaries/rouge_scores.csv"
            logger.info("Writing Rouge Scores {} to file: {}".format(scores, df_file))
            metric_df = pd.read_csv(df_file)
            new_row = {
                "model": peft_full_name,
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "rougeL": scores["rougeL"],
                "rougeLsum": scores["rougeLsum"]
            }
        elif metric_name == "bertscore":
            df_file = "summaries/bertscore_scores.csv"
            logger.info("Writing BertScore Scores {} to file: {}".format(scores, df_file))
            metric_df = pd.read_csv(df_file)
            new_row = {
                "model": peft_full_name,
                "precision_mean": scores["precision"]["mean"],
                "precision_median": scores["precision"]["median"],
                "recall_mean": scores["recall"]["mean"],
                "recall_median": scores["recall"]["median"],
                "f1_mean": scores["f1"]["mean"],
                "f1_median": scores["f1"]["median"]
            }
        elif metric_name == "bleu":
            df_file = "summaries/bleu_scores.csv"
            logger.info("Writing Bleu Scores {} to file: {}".format(scores, df_file))
            metric_df = pd.read_csv(df_file)
            new_row = {
                "model": peft_full_name,
                "bleu": scores["bleu"],
                "precisions": scores["precisions"],
                "brevity_penalty": scores["brevity_penalty"],
                "length_ratio": scores["length_ratio"],
                "translation_length": scores["translation_length"],
                "reference_length": scores["reference_length"]
            }
        # elif metric_name == "bleurt":
        #     logger.info("Writing Bleurt Scores {} to file: summaries/bleurt_scores.csv".format(scores))
        #     metric_df = pd.read_csv("summaries/bleurt_scores.csv")
        #     new_row = {
        #         "model": peft_full_name,
        #         "mean": scores["scores"]["mean"],
        #         "median": scores["scores"]["median"],
        #     }

        if peft_full_name in metric_df["model"].values:
            # Update the existing row
            metric_df.loc[metric_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
        else:
            # Add a new row
            metric_df = pd.concat([metric_df, pd.DataFrame([new_row])], ignore_index=True)

        metric_df.to_csv(df_file, index=False)

        logger.info("\n\n\nSummaries with {} Score {} saved to file {}!!!!".format(metric_name, scores,
                                                                                   test_summaries_file_name))


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and model (domain) for training")

    parser.add_argument("--checkpoint",type=str, default=None, help="Path of the PT Model Checkpoint to be loaded." )
    parser.add_argument("--trained_peft_path", type=str, help="Path of the PEFT to be loaded.")
    parser.add_argument("--training_samples", type=int, default=1, help="Number of training Samples")
    parser.add_argument("--eval_samples", type=int, default=1, help="Number of Evaluation Samples")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of Samples to be tested")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Torch Data Type to be used")
    parser.add_argument("--metric", type=str, required=True, choices=["rouge", "bertscore", "bleu", "bleurt", "all"],
                        help="Metric to be used for testing, pass 'all' if you want test on all")
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize the model")
    parser.add_argument("--sorted_dataset", type=bool, default=False, help="do you want to sort the dataset?")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    parser.add_argument("--mlm", type=bool, default=False, help="Using attention mask")
    parser.add_argument("--overwrite_results", type=bool, default=False, help="Overwrite the results generated")

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
    overwrite_results = args.overwrite_results
    # provider = "hf" if "hf" in trained_peft_path else "ah"

    peft_path_splits = trained_peft_path.split("/")
    # if peft_path_splits[0] == "results":
    #     peft_dir = "_".join(peft_path_splits)
    #     domain = peft_path_splits[3].split("_")[0]
    #     adapter_name = peft_path_splits[3]

    # elif trained_peft_path.split("/")[0] ==  :# "saved_models":
    peft_dir = peft_path_splits[1]
    domain, dataset_name, peft_type = tuple(peft_dir.split("_")[:3])
    adapter_name = "{}_{}_{}".format(domain, dataset_name, peft_type)

    provider = "hf"

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
                  test_summaries_file_name=None, overwrite_results=overwrite_results)
    wnb_run.finish()
