import argparse
import logging
import os
import statistics
import json

import adapters
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from dataset_lib import SumDataLoader, DEFAULT_DOMAIN_PROMPT, DEFAULT_SYSTEM_PROMPT, SumDomains, datasets_info_dict
from utils import generate_summary, rouge_metric, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, torch_dtypes_dict, WandBLogger, check_and_return_df, bertscore_metric, \
    bleu_metric, bleurt_metric, read_yaml, meteor_metric, power_set


# def unseen_test_data_inference(llama_model, llama_tokenizer, data_class, peft_full_name, device, logger, chat_template,
#                                col_name, metric_name):
#     if metric_name == "rouge":
#         metric = rouge_metric()
#     elif metric_name == "bertscore":
#         metric = bertscore_metric()
#     elif metric_name == "bleu":
#         metric = bleu_metric()
#     # elif metric_name == "bleurt":
#     #     metric = bleurt_metric()
#     elif metric_name == "meteor":
#         metric = meteor_metric()
#     elif metric_name == "all":
#         rouge = rouge_metric()
#         bertscore = bertscore_metric()
#         bleu = bleu_metric()
#         meteor = meteor_metric()
#         # bleurt = bleurt_metric()
#     else:
#         raise ValueError("Invalid Metric")
#
#     save_df, file_exists = True, False
#     test_summaries_file_name = "summaries/summaries_{}_{}_25samples.csv".format(data_class.domain.name.lower(), data_class.name.lower()) if "multiple" in peft_full_name else "summaries/summaries_{}_{}.csv".format(data_class.domain.name.lower(), data_class.name.lower())
#     data = pd.read_csv(data_class.local_path)
#     min_samples = len(data) #.num_rows
#
#
#
#     # if test_summaries_file_name is None:
#     #     test_summaries_file_name = "summaries/summaries_{}_{}_{}samples.csv".format(data.domain.name.lower(),
#     #                                                                                 data.dataset_name.lower(),
#     #                                                                                 min_samples)
#
#     # random_text = """
#     #         Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
#     #     """.strip()
#     #
#     # summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_text, device=device,
#     #                         chat_template=chat_template, prompt=DEFAULT_SYSTEM_PROMPT)
#     # TODO: write the testing function with a metric.
#     test_summaries = {
#         # "article": [],
#         # "truth": [],
#         col_name: []
#     }
#     df_sum, file_exists = check_and_return_df(test_summaries_file_name)
#     # col_name = col_name + "_shortprompt"
#     if col_name not in df_sum.columns:
#         logger.info("PROMPT in USE for Testing: \n'{}'".format(DEFAULT_DOMAIN_PROMPT[data_class.name.upper()]))
#         articles = df_sum["article"] if "multiple" in peft_full_name else data["content"]
#         logger.info("Running infernce of {} on {} articles.".format(peft_full_name, len(articles)))
#         i = 0
#         for i, art in enumerate(articles):
#             logger.info("Summary for {} sample".format(i+1))
#             summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=art,
#                                        device=device,
#                                        chat_template=chat_template, prompt=DEFAULT_DOMAIN_PROMPT[data_class.name.upper()])
#             test_summaries[col_name].append(summary)
#             del summary
#     else:
#         logger.info("Summaries in col: {} already exists in file: {}".format(col_name, test_summaries_file_name))
#         test_summaries[col_name] = df_sum[col_name]
#         save_df = False
#
#     scores, rouge_scores, bertscore_scores, bleu_scores, bleurt_scores, meteor_scores = 0, 0, 0, 0, 0, 0
#     truth = df_sum["truth"] if "multiple" in peft_full_name else data["summary"]
#         # metric = rouge_metric()
#     if metric_name == "all":
#         rouge_scores = rouge.compute(predictions=test_summaries[col_name], references=truth)
#         bertscore_scores = bertscore.compute(predictions=test_summaries[col_name],
#                                              references=truth, lang="en", verbose=True)
#
#         bertscore_scores["precision"] = {"mean": statistics.mean(bertscore_scores["precision"]),
#                                          "median": statistics.median(bertscore_scores["precision"])}
#         bertscore_scores["recall"] = {"mean": statistics.mean(bertscore_scores["recall"]),
#                                       "median": statistics.median(bertscore_scores["recall"])}
#         bertscore_scores["f1"] = {"mean": statistics.mean(bertscore_scores["f1"]),
#                                   "median": statistics.median(bertscore_scores["f1"])}
#
#         bleu_scores = bleu.compute(predictions=test_summaries[col_name], references=truth)
#         meteor_scores = meteor.compute(predictions=test_summaries[col_name], references=truth)
#         # bleurt_scores = bleurt.compute(predictions=test_summaries[col_name], references=truth)
#         # bleurt_scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
#         #                     "median": statistics.median(bleurt_scores["scores"])}
#
#         logger.info("ROUGE Scores: {}".format(rouge_scores))
#         logger.info("BERTSCORE Scores: {}".format(bertscore_scores))
#         logger.info("BLEU Scores: {}".format(bleu_scores))
#         logger.info("METEOR Scores: {}".format(meteor_scores))
#         # logger.info("BLEURT Scores: {}".format(bleurt_scores))
#     else:
#         if metric_name == "bertscore":
#             bertscore_scores = metric.compute(predictions=test_summaries[col_name], references=truth,
#                                     lang="en", verbose=True)
#             scores = {}
#             scores["precision"] = {"mean": statistics.mean(bertscore_scores["precision"]),
#                                    "median": statistics.median(bertscore_scores["precision"])}
#             scores["recall"] = {"mean": statistics.mean(bertscore_scores["recall"]),
#                                 "median": statistics.median(bertscore_scores["recall"])}
#             scores["f1"] = {"mean": statistics.mean(bertscore_scores["f1"]),
#                             "median": statistics.median(bertscore_scores["f1"])}
#         # elif metric_name == "bleurt":
#         #     bleurt_scores = metric.compute(predictions=test_summaries[col_name], references=truth)
#         #     scores = {}
#         #     scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
#         #                         "median": statistics.median(bleurt_scores["scores"])}
#         else:
#             scores = metric.compute(predictions=test_summaries[col_name], references=truth)
#         logger.info("{} Scores: {}".format(metric_name, scores))
#
#     if file_exists:
#         test_summaries.pop("content", None)
#         test_summaries.pop("summary", None)
#     else:
#         df_sum["article"] = data["content"]
#         df_sum["truth"] = truth
#     df_sum[col_name] = test_summaries[col_name]
#
#     if save_df:
#         df_sum.to_csv(test_summaries_file_name, index=False)
#
#     # TODO: Write Scores to a CSV file directly, without storing it in a txt file.
#     if metric_name == "all":
#         from datetime import datetime
#         # with open("summaries/rouge_scores.txt", "a") as fp:
#         #     fp.write("[{}] Summaries of {} for {} samples has rouge Scores \n {} \n\n".format(datetime.today().date(),
#         #                                                                                       peft_full_name,
#         #                                                                                       min_samples,
#         #                                                                                       rouge_scores))
#
#         # ROUGE
#         f_name = "summaries/unseen_data_rouge_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_rouge_scores.csv"
#         logger.info("Writing ROUGE Scores {} to file: {}".format(rouge_scores, f_name))
#         try:
#             rouge_df = pd.read_csv(f_name)
#         except Exception as e:
#             rouge_df = pd.DataFrame(columns=["model", "rouge1", "rouge2", "rougeL", "rougeLsum"])
#
#         new_row = {
#             "model": peft_full_name,
#             "rouge1": rouge_scores["rouge1"],
#             "rouge2": rouge_scores["rouge2"],
#             "rougeL": rouge_scores["rougeL"],
#             "rougeLsum": rouge_scores["rougeLsum"]
#         }
#         if peft_full_name in rouge_df["model"].values:
#             # Update the existing row
#             rouge_df.loc[rouge_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
#         else:
#             # Add a new row
#             rouge_df = pd.concat([rouge_df, pd.DataFrame([new_row])], ignore_index=True)
#         rouge_df.to_csv(f_name, index=False)
#
#         logger.info("\n\n\nSummaries with rouge Score {} saved to file {}!!!!".format(rouge_scores,
#                                                                                       test_summaries_file_name))
#
#         # with open("summaries/bertscore_scores.txt", "a") as fp:
#         #     fp.write("[{}] Summaries of {} for {} samples has bertscore Scores \n {} \n\n".format(datetime.today().date(),
#         #                                                                                           peft_full_name,
#         #                                                                                           min_samples,
#         #                                                                                           bertscore_scores))
#         # BERTSCORE
#         f_name = "summaries/unseen_data_bertscore_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_bertscore_scores.csv"
#         try:
#             bertscore_df = pd.read_csv(f_name)
#         except Exception as e:
#             bertscore_df = pd.DataFrame(columns=["model", "precision_mean", "precision_median", "recall_mean",
#                                                  "recall_median", "f1_mean", "f1_median"])
#
#         logger.info("Writing BERTScores {} to file: {}".format(bertscore_scores, f_name))
#         new_row = {
#             "model": peft_full_name,
#             "precision_mean": bertscore_scores["precision"]["mean"],
#             "precision_median": bertscore_scores["precision"]["median"],
#             "recall_mean": bertscore_scores["recall"]["mean"],
#             "recall_median": bertscore_scores["recall"]["median"],
#             "f1_mean": bertscore_scores["f1"]["mean"],
#             "f1_median": bertscore_scores["f1"]["median"]
#         }
#         if peft_full_name in bertscore_df["model"].values:
#             # Update the existing row
#             bertscore_df.loc[bertscore_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
#         else:
#             # Add a new row
#             bertscore_df = pd.concat([bertscore_df, pd.DataFrame([new_row])], ignore_index=True)
#         bertscore_df.to_csv(f_name, index=False)
#
#         # TODO: Add the scores to the bertscore_scores.csv file
#         logger.info("\n\n\nSummaries with bertscore Score {} saved to file {}!!!!".format(bertscore_scores,
#                                                                                           test_summaries_file_name))
#
#         # BLEU
#         # with open("summaries/bleu_scores.txt", "a") as fp:
#         #     fp.write("[{}] Summaries of {} for {} samples has bleu Scores \n {} \n\n".format(datetime.today().date(),
#         #                                                                                      peft_full_name, min_samples,
#         #                                                                                      bleu_scores))
#         f_name = "summaries/unseen_data_bleu_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_bleu_scores.csv"
#         logger.info("Writing Bleu Scores {} to file: {}".format(bleu_scores, f_name))
#         try:
#             bleu_df = pd.read_csv(f_name)
#         except Exception as e:
#             bleu_df = pd.DataFrame(columns=["model", "bleu", "precisions", "brevity_penalty", "length_ratio",
#                                             "translation_length", "reference_length"])
#         new_row = {
#             "model": peft_full_name,
#             "bleu": bleu_scores["bleu"],
#             "precisions": json.dumps(bleu_scores["precisions"]), # bleu_scores["precisions"],
#             "brevity_penalty": bleu_scores["brevity_penalty"],
#             "length_ratio": bleu_scores["length_ratio"],
#             "translation_length": bleu_scores["translation_length"],
#             "reference_length": bleu_scores["reference_length"]
#         }
#         if peft_full_name in bleu_df["model"].values:
#             # Update the existing row
#             bleu_df.loc[bleu_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
#         else:
#             # Add a new row
#             bleu_df = pd.concat([bleu_df, pd.DataFrame([new_row])], ignore_index=True)
#         bleu_df.to_csv(f_name, index=False)
#         logger.info("\n\n\nSummaries with bleu Score {} saved to file {}!!!!".format(bleu_scores,
#                                                                                      test_summaries_file_name))
#
#         # METEOR
#         f_name = "summaries/unseen_data_meteor_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_meteor_scores.csv"
#         logger.info("Writing Meteor Scores {} to file: {}".format(meteor_scores, f_name))
#         try:
#             meteor_df = pd.read_csv(f_name)
#         except Exception as e:
#             meteor_df = pd.DataFrame(columns=["model", "meteor"])
#         new_row = {
#             "model": peft_full_name,
#             "meteor": meteor_scores["meteor"]
#         }
#         if peft_full_name in meteor_df["model"].values:
#             # Update the existing row
#             meteor_df.loc[meteor_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
#         else:
#             # Add a new row
#             meteor_df = pd.concat([meteor_df, pd.DataFrame([new_row])], ignore_index=True)
#         meteor_df.to_csv(f_name, index=False)
#         logger.info("\n\n\nSummaries with meteor Score {} saved to file {}!!!!".format(meteor_scores,
#                                                                                        test_summaries_file_name))
#
#         # TODO: Add the scores to the bleu_scores.csv file
#         # BLEURT
#         # with open("summaries/bleurt_scores.txt", "a") as fp:
#         #     fp.write("[{}] Summaries of {} for {} samples has bleuRT Scores \n {} \n\n".format(datetime.today().date(),
#         #                                                                                      peft_full_name, min_samples,
#         #                                                                                      bleurt_scores))
#
#         # TODO: Uncomment After fixing
#         # logger.info("Writing BleuRT Scores {} to file: summaries/bleurt_scores.csv".format(bleurt_scores))
#         # bleurt_df = pd.read_csv("summaries/bleurt_scores.csv")
#         # new_row = {
#         #     "model": peft_full_name,
#         #     "mean": bleurt_scores["scores"]["mean"],
#         #     "median": bleurt_scores["scores"]["median"],
#         # }
#         # if peft_full_name in bleurt_df["model"].values:
#         #     # Update the existing row
#         #     bleurt_df.loc[bleurt_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
#         # else:
#         #     # Add a new row
#         #     bleurt_df = pd.concat([bleurt_df, pd.DataFrame([new_row])], ignore_index=True)
#         # bleurt_df.to_csv("summaries/bleurt_scores.csv", index=False)
#         # logger.info("\n\n\nSummaries with bleuRT Score {} saved to file {}!!!!".format(scores,
#         #                                                                              test_summaries_file_name))
#
#     else:
#         # with open("summaries/{}_scores.txt".format(metric_name), "a") as fp:
#         #     from datetime import datetime
#         #     fp.write("[{}] Summaries of {} for {} samples has {} Scores \n {} \n\n".format(datetime.today().date(),
#         #                                                                                    peft_full_name, min_samples,
#         #                                                                                    metric_name, scores))
#         if metric_name == "rouge":
#             df_file = "summaries/unseen_data_rouge_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_rouge_scores.csv"
#             logger.info("Writing Rouge Scores {} to file: {}".format(scores, df_file))
#             try:
#                 metric_df = pd.read_csv(df_file)
#             except Exception as e:
#                 metric_df = pd.DataFrame(columns=["model", "rouge1", "rouge2", "rougeL", "rougeLsum"])
#             new_row = {
#                 "model": peft_full_name,
#                 "rouge1": scores["rouge1"],
#                 "rouge2": scores["rouge2"],
#                 "rougeL": scores["rougeL"],
#                 "rougeLsum": scores["rougeLsum"]
#             }
#         elif metric_name == "bertscore":
#             df_file = "summaries/unseen_data_bertscore_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_bertscore_scores.csv"
#             logger.info("Writing BertScore Scores {} to file: {}".format(scores, df_file))
#             try:
#                 metric_df = pd.read_csv(df_file)
#             except Exception as e:
#                 metric_df = pd.DataFrame(columns=["model", "precision_mean", "precision_median", "recall_mean",
#                                                  "recall_median", "f1_mean", "f1_median"])
#             new_row = {
#                 "model": peft_full_name,
#                 "precision_mean": scores["precision"]["mean"],
#                 "precision_median": scores["precision"]["median"],
#                 "recall_mean": scores["recall"]["mean"],
#                 "recall_median": scores["recall"]["median"],
#                 "f1_mean": scores["f1"]["mean"],
#                 "f1_median": scores["f1"]["median"]
#             }
#         elif metric_name == "bleu":
#             df_file = "summaries/unseen_data_bleu_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_bleu_scores.csv"
#             logger.info("Writing Bleu Scores {} to file: {}".format(scores, df_file))
#             try:
#                 metric_df = pd.read_csv(df_file)
#             except Exception as e:
#                 metric_df = pd.DataFrame(columns=["model", "bleu", "precisions", "brevity_penalty", "length_ratio",
#                                                  "translation_length", "reference_length"])
#             new_row = {
#                 "model": peft_full_name,
#                 "bleu": scores["bleu"],
#                 "precisions": json.dumps(scores["precisions"]), #scores["precisions"],
#                 "brevity_penalty": scores["brevity_penalty"],
#                 "length_ratio": scores["length_ratio"],
#                 "translation_length": scores["translation_length"],
#                 "reference_length": scores["reference_length"]
#             }
#         elif metric_name == "meteor":
#             df_file = "summaries/unseen_data_meteor_scores25.csv" if "multiple" in peft_full_name else "summaries/unseen_data_meteor_scores.csv"
#             logger.info("Writing Meteor Scores {} to file: {}".format(scores, df_file))
#             try:
#                 metric_df = pd.read_csv(df_file)
#             except Exception as e:
#                 metric_df = pd.DataFrame(columns=["model", "meteor"])
#             new_row = {
#                 "model": peft_full_name,
#                 "meteor": scores["meteor"]
#             }
#         # elif metric_name == "bleurt":
#         #     logger.info("Writing Bleurt Scores {} to file: summaries/bleurt_scores.csv".format(scores))
#         #     metric_df = pd.read_csv("summaries/bleurt_scores.csv")
#         #     new_row = {
#         #         "model": peft_full_name,
#         #         "mean": scores["scores"]["mean"],
#         #         "median": scores["scores"]["median"],
#         #     }
#
#         if peft_full_name in metric_df["model"].values:
#             # Update the existing row
#             metric_df.loc[metric_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
#         else:
#             # Add a new row
#             metric_df = pd.concat([metric_df, pd.DataFrame([new_row])], ignore_index=True)
#
#         metric_df.to_csv(df_file, index=False)
#
#         logger.info("\n\n\nSummaries with {} Score {} saved to file {}!!!!".format(metric_name, scores,
#                                                                                    test_summaries_file_name))


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--config_pefts_file_path", default=None, type=str, help="Path of the config file containing pefts and dataset for multiple pefts inference")
    parser.add_argument("--peft_path", type=str, default=None, help="For single peft")
    parser.add_argument("--peft_dir", type=str, default="trained_pefts/", help="Storage directory")
    parser.add_argument("--test_dataset", default=None, type=str, help="Only test dataest.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Torch Data Type to be used")
    parser.add_argument("--metric", type=str, required=True, choices=["rouge", "bertscore", "bleu", "bleurt", "all"],
                        help="Metric to be used for testing, pass 'all' if you want test on all")
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize the model")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    parser.add_argument("--mlm", type=bool, default=False, help="Using attention mask")

    main_directory, config_file = "", ""
    configs = {}

    args = parser.parse_args()
    if args.config_pefts_file_path is None and args.peft_path is None:
        raise ValueError("Please provide the path of the config file containing pefts and dataset for multiple pefts inference or the path of the peft for single peft inference")
    if args.config_pefts_file_path is not None:
        config_file = main_directory + args.config_pefts_file_path
        configs = read_yaml(file_name=config_file)
    elif args.peft_path is not None:
        configs["pefts"] = [args.peft_path]
    mlm = True if args.mlm else False
    chat_template = args.chat_template
    quantize = args.quantize
    metric = args.metric
    peft_dir = args.peft_dir
    torch_dtype = torch_dtypes_dict[args.torch_dtype]
    
    use_instruct_model = True # if "instruct" in trained_peft_path or args.chat_template else False
    # provider = "hf" if "hf" in trained_peft_path else "ah"
    
    dataset_name = args.test_dataset if args.test_dataset else configs["test_dataset"]

    power_sets = power_set(configs["pefts"])

    print("Power Sets: ", power_sets)

    for i, peft_set in enumerate(power_sets):
        if len(peft_set) < 2:
            continue
        configs["pefts"] = list(peft_set)


        # elif trained_peft_path.split("/")[0] ==  :# "saved_models":
        peft_names = []
        # if not zero_shot:
        zero_shot = False
        for i, path in enumerate(configs["pefts"]):
            a_name = path.split("_checkpoint")[0]
            configs["pefts"][i] = configs["pefts"][i] + "/" + a_name
            peft_names.append(a_name)

        provider = "hf"

        load_dotenv(".env")
        hf_token = os.getenv("HF_TOKEN")
        wandb_api_key = os.getenv("WANDB_API_KEY")
        run_name, wnb_run, logger, console_handler = None, None, None, None
        # run_name = 'unseen_data_inference_{}_domain_{}_{}.log'.format("cross" if "cross" in config_file else "within",
        #                                                               ("-".join(peft_names) if len(peft_names) > 1 else
        #                                                                peft_names[0]) if not zero_shot else "zero_shot",
        #                                                               dataset_name)

        run_name = 'holdout_data_inference_{}_domain_{}_{}.log'.format("cross" if "cross" in config_file else "within",
                                                                      ("-".join(peft_names) if len(peft_names) > 1 else
                                                                       peft_names[0]) if not zero_shot else "zero_shot",
                                                                      dataset_name)

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

        logger.info("Configs file path: {}".format(config_file))
        logger.info("Device in use: {}".format(device))
        llama = None
        # llama_model = get_pretrained_model(ah=ah)
        llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize, mlm=mlm, torch_dtype=torch_dtype)
        # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

        # logger.info("Check point MODEL: \n{}".format(llama.model))

        # Method 1 - HuggingFace
        logger.info(" -->{}".format(configs))
        if not zero_shot:
            for a_path, a_name in zip(configs["pefts"], peft_names):
                llama.model.load_adapter(peft_dir+a_path, adapter_name=a_name)
        # llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
        # llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name=adapter_name,
        #                                                           torch_dtype=torch_dtype)
            logger.info("Active Adapters in Model before enabling adapters: {}".format(llama.model.active_adapters()))
            llama.model.set_adapter(peft_names)
            llama.model.enable_adapters()
            logger.info("Active Adapters in Model after enabling adapters: {}".format(llama.model.active_adapters()))

            # llama.model.set_adapter([peft_names])
            # logger.info("Active Adapters in Model: {}".format(llama.model.active_adapters()))
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
        from inference_unseen_data import unseen_test_data_inference
        #TODO: do better nomenclature for the peft_full_name and col_name
        unseen_test_data_inference(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data_class=data_class,
                                   # peft_full_name=("multiple_"+"-".join(peft_names)+f"_{dataset_name}" if
                                   #                 len(peft_names) > 1 else peft_names[0]+f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   peft_full_name=("within_domain_test_" + "-".join(peft_names) + f"_{dataset_name}" if
                                                   len(peft_names) > 1 else peft_names[
                                                                                0] + f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   # col_name=("multiple_"+"-".join(peft_names)+f"_{dataset_name}" if
                                   #           len(peft_names) > 1 else peft_names[0]+f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   col_name=("within_domain_test_" + "-".join(peft_names) + f"_{dataset_name}" if
                                             len(peft_names) > 1 else peft_names[
                                                                          0] + f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   logger=logger, device=device, chat_template=chat_template, metric_name=metric)
        wnb_run.finish()
