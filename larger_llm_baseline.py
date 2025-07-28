import json
import logging
import os

import argparse
import statistics

import openai
import pandas as pd
import transformers

import wandb
from openai import OpenAI
from together import Together

from dataset_lib import datasets_info_dict, SumDomains, DEFAULT_DOMAIN_PROMPT
from utils import rouge_metric, bertscore_metric, bleu_metric, meteor_metric, WandBLogger, check_and_return_df

# together: fde6a03877504ef0eeccd9a67c867cb15f6be6586ac080b4f1ff3d1f1d414d6c; df491a9f1566b190b9ed8eda715fd1f32f15d89b73317746867599b305bad7f9; 55cb14a8e643601f56c6b0956ecef4aa7970f67a48feac81e101ad8637b06a34
# openai: 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
# DEFAULT_DOMAIN_PROMPT = {
#     "scientific": """
#         Summarize the provided scientific article in a clear and concise paragraph. Include the study's objective, 
#         background, methodology, key findings, and conclusions, ensuring the summary represents the article's essence.""".strip(),

#     "MEDICAL".lower(): """
#         Provide a cohesive summary of the given medical article in one paragraph. Highlight the study's objective, 
#         background, methods, major conclusions, and potential clinical implications in a clear and professional manner.""".strip(),

#     "LEGAL".lower(): """
#         Generate a concise paragraph summarizing the provided legal case study. Address the background, legal questions, 
#         key arguments, rulings, and any significant precedents, ensuring clarity and accuracy.""".strip(),

#     "NEWS".lower(): """
#         Create a clear and concise summary of the given news article in one paragraph. Focus on the main event, its 
#         context, key details, and broader implications, presenting the information in an informative manner.""".strip()
# }

# def make_prompt(article, summs):
def call_gpt4(client_openai, model, role, content):
    # try:
    # Using OpenAI's ChatCompletion endpoint with GPT-4
    # response = client_openai.ChatCompletion.create(
    response = client_openai.chat.completions.create(
        model=model,  # "gpt-4",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": content}
        ],
        temperature=0.7,
        max_tokens=256,
        logprobs=True,  # Needs to be True if num_log_probs > 0
        top_logprobs=0,
    )

    # Extract the content from the response
    # result = response['choices'][0]['message']['content'].strip()
    result = response.choices[0].message.content

    # Attempt to parse the JSON output if provided
    try:
        parsed_result = json.loads(result)
        return parsed_result
    except json.JSONDecodeError:
        # If the output isn't valid JSON, just return the raw text
        print("Warning: The response was not valid JSON. Returning raw text.")
        return result

    # except Exception as e:
    # print(f"Error: {e}")
    # return None


model_dict = {
    "gpt": "gpt-4-turbo",
    "mixtral": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "llama3.3": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "llama3.1": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
}


def tokenize_and_join(prompt, article, token_len=8192):

    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    article_tokens = tokenizer(article, return_tensors="pt")
    if len(prompt_tokens["input_ids"][0]) + len(article_tokens["input_ids"][0]) > token_len:
        ll = token_len - len(prompt_tokens["input_ids"][0])-2
        truncate_article_tokens = article_tokens["input_ids"][0][1:ll]
        # truncate_article_tokens.append(tokenizer.eos_token_id)
        truncated_article = tokenizer.decode(truncate_article_tokens)
    else:
        truncated_article = article
    tt1 = tokenizer(truncated_article, return_tensors="pt")
    print("Total TOkens ====  ", len(prompt_tokens["input_ids"][0]), len(tt1["input_ids"][0]), len(prompt_tokens["input_ids"][0])+len(tt1["input_ids"][0]))

    return truncated_article


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")
    parser.add_argument("--platform", type=str, choices=["openai", "together"], required=True, help="Platform")
    # parser.add_argument("--task", type=str, choices=["w0", "wo0"], required=True, help="Platform")
    # parser.add_argument("remove_key", type=str, help="Key to remove from the data")
    # w0: with zero and without within, wo0: without zeor with within
    args = parser.parse_args()
    platform = args.platform

    run_name = 'llama3_70B_instruct_zero_shot_learning_unseen_domain_datasets_all_samples.log'

    logging.basicConfig(
        filename='logs/{}'.format(run_name),
        # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    wnb_run = wandb.init(name=run_name)
    # device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    logger = logging.getLogger()
    wnb = WandBLogger()
    wnb.wandb = wandb
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the log level for the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(wnb)

    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    client = Together(api_key="4b382feef4efb3d2ca5ac493dde37f45cb9c24946eb48ab79ce11f9163efbfb9")
    # model = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"  # "llama3.1" #"mixtral" #"mistralai/Mixtral-8x22B-Instruct-v0.1"
    model = "meta-llama/Llama-3-70b-chat-hf"
    rouge = rouge_metric()
    bertscore = bertscore_metric()
    bleu = bleu_metric()
    meteor = meteor_metric()
        # bleurt = bleurt_metric()
    # else:
    #     raise ValueError("Invalid Metric")

    save_df, file_exists = True, False

    unseen_data = datasets_info_dict[SumDomains.UNSEEN_TEST]

    for dataset_name in ["medical"]: #unseen_data.keys():
        data_class = unseen_data[dataset_name]
        test_summaries_file_name = "summaries/summaries_{}_{}.csv".format(data_class.domain.name.lower(), data_class.name.lower())
        if data_class.name.lower() in ["medical", "legal"]:
            test_summaries_file_name = test_summaries_file_name.replace(".csv", ".xlsx")
        data = pd.read_excel(data_class.local_path) if ".xlsx" in data_class.local_path else pd.read_csv(
            data_class.local_path)
        logger.info("Original dataset len: {}".format(len(data)))  # .num_rows
        col_name = "zero_shot_llama3_70B_instruct_512"
        peft_full_name = col_name + "_" + dataset_name
        test_summaries = {
            # "article": [],
            # "truth": [],
            col_name: []
        }
        logger.info("Dataset: {} in file {}.".format(data_class.name, test_summaries_file_name))
        df_sum, file_exists = check_and_return_df(test_summaries_file_name)
        if col_name not in df_sum.columns:
            logger.info("PROMPT in USE for Testing: \n'{}'".format(DEFAULT_DOMAIN_PROMPT[data_class.name.upper()]))
            # articles = df_sum["article"] if "multiple" in peft_full_name or "25" in test_summaries_file_name or not df_sum.empty else data["content"]
            articles = df_sum["article"] if not df_sum.empty else data["content"]
            logger.info("Running infernce of {} on {} articles.".format(peft_full_name, len(articles)))
            i = 0
            for i, art in enumerate(articles):
                logger.info("Summary for {} sample".format(i + 1))
                # summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=art, kshot=kshot,
                #                            samples=samples, device=device,
                #                            chat_template=chat_template,
                #                            prompt=DEFAULT_DOMAIN_PROMPT[data_class.name.upper()])
                # GEN summary function:
                if data_class.name.lower() == "medical":
                    new_art = tokenize_and_join(DEFAULT_DOMAIN_PROMPT[data_class.name.upper()], art,
                                                token_len=8192 - 514 - 20)
                else:
                    new_art = tokenize_and_join(DEFAULT_DOMAIN_PROMPT[data_class.name.upper()], art, token_len=8192-514)
                response = client.chat.completions.create(
                    model=model,
                    # messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
                    messages=[
                        {"role": "system", "content": DEFAULT_DOMAIN_PROMPT[data_class.name.upper()]},
                        {"role": "user", "content": f"Article:{new_art}"},
                    ],
                    temperature=0.3,
                    max_tokens=512,
                )
                summary = response.choices[0].message.content
                test_summaries[col_name].append(summary)
                del summary
        else:
            logger.info("Summaries in col: {} already exists in file: {}".format(col_name, test_summaries_file_name))
            test_summaries[col_name] = df_sum[col_name]
            save_df = False

            # # TODO: RunAgain with Together
            # if platform == "together":
            #     response = client.chat.completions.create(
            #         model=model_dict[model],
            #         # messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
            #         messages=[
            #             {"role": "system", "content": role},
            #             {"role": "user", "content": content},
            #         ],
            #         temperature=0.7,
            #         max_tokens=512,
            #     )
            #     resp = response.choices[0].message.content

        scores, rouge_scores, bertscore_scores, bleu_scores, bleurt_scores, meteor_scores = 0, 0, 0, 0, 0, 0
        # truth = df_sum["truth"] if "multiple" in peft_full_name or "25" in test_summaries_file_name else data["summary"]
        truth = df_sum["truth"] if not df_sum.empty else data["summary"]
        # metric = rouge_metric()
        # if metric_name == "all":
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
        meteor_scores = meteor.compute(predictions=test_summaries[col_name], references=truth)
        # bleurt_scores = bleurt.compute(predictions=test_summaries[col_name], references=truth)
        # bleurt_scores["scores"] = {"mean": statistics.mean(bleurt_scores["scores"]),
        #                     "median": statistics.median(bleurt_scores["scores"])}

        logger.info("ROUGE Scores: {}".format(rouge_scores))
        logger.info("BERTSCORE Scores: {}".format(bertscore_scores))
        logger.info("BLEU Scores: {}".format(bleu_scores))
        logger.info("METEOR Scores: {}".format(meteor_scores))
        # logger.info("BLEURT Scores: {}".format(bleurt_scores))

        if file_exists:
            test_summaries.pop("content", None)
            test_summaries.pop("summary", None)
        else:
            df_sum["article"] = data["content"]
            df_sum["truth"] = truth
        df_sum[col_name] = test_summaries[col_name]

        if save_df:
            if ".xlsx" in test_summaries_file_name:
                from utils import clean_illegal_chars

                df_sum[col_name] = df_sum[col_name].apply(clean_illegal_chars)
                df_sum = df_sum.applymap(clean_illegal_chars)
                df_sum.to_excel(test_summaries_file_name, index=False)
            else:
                df_sum.to_csv(test_summaries_file_name, index=False)

        # TODO: Write Scores to a CSV file directly, without storing it in a txt file.
        # if metric_name == "all":

        # ROUGE
        f_name = "summaries/unseen_data_rouge_scores25.csv" if "multiple" in peft_full_name or "25" in test_summaries_file_name else "summaries/unseen_data_rouge_scores.csv"
        logger.info("Writing ROUGE Scores {} to file: {}".format(rouge_scores, f_name))
        try:
            rouge_df = pd.read_csv(f_name)
        except Exception as e:
            rouge_df = pd.DataFrame(columns=["model", "rouge1", "rouge2", "rougeL", "rougeLsum"])

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
        rouge_df.to_csv(f_name, index=False)

        logger.info("\n\n\nSummaries with rouge Score {} saved to file {}!!!!".format(rouge_scores,
                                                                                      test_summaries_file_name))

        # with open("summaries/bertscore_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has bertscore Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                           peft_full_name,
        #                                                                                           min_samples,
        #                                                                                           bertscore_scores))
        # BERTSCORE
        f_name = "summaries/unseen_data_bertscore_scores25.csv" if "multiple" in peft_full_name or "25" in test_summaries_file_name else "summaries/unseen_data_bertscore_scores.csv"
        try:
            bertscore_df = pd.read_csv(f_name)
        except Exception as e:
            bertscore_df = pd.DataFrame(columns=["model", "precision_mean", "precision_median", "recall_mean",
                                                 "recall_median", "f1_mean", "f1_median"])

        logger.info("Writing BERTScores {} to file: {}".format(bertscore_scores, f_name))
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
        bertscore_df.to_csv(f_name, index=False)

        # TODO: Add the scores to the bertscore_scores.csv file
        logger.info("\n\n\nSummaries with bertscore Score {} saved to file {}!!!!".format(bertscore_scores,
                                                                                          test_summaries_file_name))

        # BLEU
        # with open("summaries/bleu_scores.txt", "a") as fp:
        #     fp.write("[{}] Summaries of {} for {} samples has bleu Scores \n {} \n\n".format(datetime.today().date(),
        #                                                                                      peft_full_name, min_samples,
        #                                                                                      bleu_scores))
        f_name = "summaries/unseen_data_bleu_scores25.csv" if "multiple" in peft_full_name or "25" in test_summaries_file_name else "summaries/unseen_data_bleu_scores.csv"
        logger.info("Writing Bleu Scores {} to file: {}".format(bleu_scores, f_name))
        try:
            bleu_df = pd.read_csv(f_name)
        except Exception as e:
            bleu_df = pd.DataFrame(columns=["model", "bleu", "precisions", "brevity_penalty", "length_ratio",
                                            "translation_length", "reference_length"])
        new_row = {
            "model": peft_full_name,
            "bleu": bleu_scores["bleu"],
            "precisions": json.dumps(bleu_scores["precisions"]),  # bleu_scores["precisions"],
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
        bleu_df.to_csv(f_name, index=False)
        logger.info("\n\n\nSummaries with bleu Score {} saved to file {}!!!!".format(bleu_scores,
                                                                                     test_summaries_file_name))

        # METEOR
        f_name = "summaries/unseen_data_meteor_scores25.csv" if "multiple" in peft_full_name or "25" in test_summaries_file_name else "summaries/unseen_data_meteor_scores.csv"
        logger.info("Writing Meteor Scores {} to file: {}".format(meteor_scores, f_name))
        try:
            meteor_df = pd.read_csv(f_name)
        except Exception as e:
            meteor_df = pd.DataFrame(columns=["model", "meteor"])
        new_row = {
            "model": peft_full_name,
            "meteor": meteor_scores["meteor"]
        }
        if peft_full_name in meteor_df["model"].values:
            # Update the existing row
            meteor_df.loc[meteor_df["model"] == peft_full_name, list(new_row.keys())] = list(new_row.values())
        else:
            # Add a new row
            meteor_df = pd.concat([meteor_df, pd.DataFrame([new_row])], ignore_index=True)
        meteor_df.to_csv(f_name, index=False)
        logger.info("\n\n\nSummaries with meteor Score {} saved to file {}!!!!".format(meteor_scores,
                                                                                       test_summaries_file_name))


