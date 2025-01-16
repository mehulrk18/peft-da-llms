import glob
import os.path

import argparse
import pandas as pd
from dotenv import load_dotenv

from FActScore.fact_score import calc_factscore
from utils.fscore_utils import df_to_jsonl_for_factscore

load_dotenv()


factscore_results_file = "summaries/factscore_results.csv"
STORE_DATA_DIR = ""

def generating_factscores_for_summaries(model_name, grounding_provided, open_ai_key, domain, dataset_name,
                                        summary_file_path=None):
    # if domain == "unseen_test":
    #     df = pd.read_csv("summaries/summaries_{}_{}.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    # else:
    #     df = pd.read_csv("summaries/summaries_{}_{}_150samples.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    df = pd.read_csv(summary_file_path, encoding="ISO-8859-1")
    pefts = df.columns[2:]

    if os.path.exists(factscore_results_file):
        fs_df = pd.read_csv(factscore_results_file)
    else:
        # fs_df = pd.DataFrame(columns=["peft_name", "score", "num_atomic_facts"])
        fs_df = pd.DataFrame(columns=["test_domain_dataset", "peft_name", "score", "num_atomic_facts"])

    print("Data Directory: ", STORE_DATA_DIR)

    fs_reslts = []

    for _peft in pefts:
        # skipping zero shot for now.
        if "zero_shot_" in _peft:
            continue
        print("Running FactScore for: Domain: {} - Dataset: {} - Peft: {}".format(domain, dataset_name, _peft))
        new_df = df[["article", "truth", _peft]]  # summary
        prediction_col_name = "data-{}_{}-peft-{}".format(domain, dataset_name, _peft)

        new_df.rename(columns={_peft: prediction_col_name}, inplace=True)

        jsonl_path = df_to_jsonl_for_factscore(df=new_df, predictions_col_name=prediction_col_name, main_data_dir=STORE_DATA_DIR)

        arg_dict = {
            "input_path": jsonl_path,
            "model_name": model_name,
            "grounding_provided": grounding_provided,
            "openai_key": open_ai_key,
        }

        result_scores = calc_factscore(arg_dict=arg_dict)
        print("FACTSCORE for {}: {}".format(prediction_col_name, result_scores), "\n\n")

        res_obj = {
            "test_domain_dataset": "{}_{}".format(domain, dataset_name),
            "peft_name": _peft,
            "score": result_scores["score"],
            "num_atomic_facts": result_scores["num_atomic_facts"]
        }
        fs_reslts.append(res_obj)

    for obj in fs_reslts:
        if obj["test_domain_dataset"] in fs_df["test_domain_dataset"].values and obj["peft_name"] in fs_df["peft_name"].values:
            # Update the row where the "Dataset" value matches
            # fs_df.loc[fs_df["peft_name"] == obj["peft_name"], obj.keys()] = obj.values()
            # fs_df.loc[fs_df["peft_name"] == obj["peft_name"] and fs_df["test_domain_data"] == obj["test_domain_data"], obj.keys()] = obj.values()
            
            
            # fs_df.loc[
            #     (fs_df["peft_name"] == obj["peft_name"]) & (fs_df["test_domain_data"] == obj["test_domain_data"]),
            #     list(obj.keys())
            # ] = list(obj.values())
            if "test_domain_dataset" not in fs_df.columns:
                fs_df["test_domain_dataset"] = None  # Add with default value if missing

            # Ensure the column exists in obj
            if "test_domain_dataset" in obj:
                fs_df.loc[
                    (fs_df["peft_name"] == obj["peft_name"]) & (fs_df["test_domain_dataset"] == obj["test_domain_dataset"]),
                    list(obj.keys())
                ] = list(obj.values())
            else:
                print("'test_domain_dataset' is missing in obj")

        else:
            # Append the new row
            fs_df = pd.concat([fs_df, pd.DataFrame([obj])], ignore_index=True)

    fs_df.to_csv(factscore_results_file, index=False)


if __name__ == "__main__":
    # Please install the requirements.txt, have a .env file with required tokens( OPENAI, huggingface), and api.key file with openai token
    model_name: str = "GPT-4o-mini"
    grounding_provided: bool = True
    openai_key = "api.key"

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")
    parser.add_argument("--data_dir", default="", type=str, help="Path to main data directory")

    # domain = "scientific"
    # dataset_name = "arxiv"
    args = parser.parse_args()
    STORE_DATA_DIR = args.data_dir

    from dataset_lib import datasets_info_dict, SumDomains

    did = datasets_info_dict.pop(SumDomains.UNSEEN_TEST)

    for domain, datasets in datasets_info_dict.items():
        for dataset_name in datasets:
            print("Calculating FactScore for: Domain: {} - Dataset: {}".format(domain.name.lower(), dataset_name))
            generating_factscores_for_summaries(model_name, grounding_provided, openai_key, domain.name.lower(), dataset_name,
                                                summary_file_path="summaries/summaries_{}_{}_150samples.csv".format(domain.name.lower(), dataset_name))

    # generating_factscores_for_summaries(model_name, grounding_provided, openai_key, domain, dataset_name,
    #                                     summary_file_path="summaries/summaries_scientific_arxiv_150samples.csv")

    # for all generated summaries for all the 14 + 4 test datasets, on all PEFTs
    # summary_files = glob.glob("summaries/summaries_*.csv")
    #
    # for summary_file in summary_files:
    #     if "unseen_test" in summary_file:
    #         splits = summary_file.split("_")
    #         domain = "unseen_test"
    #         dataset_name = splits[3]
    #     else:
    #         domain, dataset_name = summary_file.split("_")[1:3]
    #     # dataset_name = summary_file.split("_")[2].split(".")[0]
    #     generating_factscores_for_summaries(model_name, grounding_provided, openai_key, domain, dataset_name,
    #                                         summary_file_path=summary_file)

    # jsonl_paths = csv_to_jsonl_for_factscore(csv_results_dir)


    # jsonl_path = df_to_jsonl_for_factscore(df=1, predictions_col_name=1)
    # # print(jsonl_paths)
    # for json_path in jsonl_paths:
    #     print("Json PATH:", json_path)
    #     arg_dict = {
    #         "input_path": json_path,
    #         "model_name": model_name,
    #         "grounding_provided": grounding_provided,
    #         "openai_key": openai_key,
    #     }
    #
    #     result_scores = calc_factscore(arg_dict=arg_dict)
    #     print("FACTSCORE:", result_scores)


    # if reading csv here:
        # df = pd.read_csv(summaries_name, encoding="ISO-8859-1")
        # keep only the columns needed for factscore -> article, summary, domain_dataset_peft
