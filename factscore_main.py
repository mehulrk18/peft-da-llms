import glob
import os.path

import argparse
import pandas as pd
from dotenv import load_dotenv

from FActScore.fact_score import calc_factscore
from utils import read_yaml
from utils.fscore_utils import df_to_jsonl_for_factscore

load_dotenv()

factscore_results_file = "summaries/factscore_results.csv"
STORE_DATA_DIR = ""


def generating_factscores_for_summaries(model_name, grounding_provided, open_ai_key, domain, dataset_name,
                                        factscore_results_file=factscore_results_file,
                                        summary_file_path=None, columns_to_skip=None):
    # if domain == "unseen_test":
    #     df = pd.read_csv("summaries/summaries_{}_{}.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    # else:
    #     df = pd.read_csv("summaries/summaries_{}_{}_150samples.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    if summary_file_path.endswith(".csv"):
        df = pd.read_csv(summary_file_path, encoding="ISO-8859-1")
    elif summary_file_path.endswith(".xlsx"):
        df = pd.read_excel(summary_file_path)
    else:
        raise ValueError("Please provide a valid file path, either .csv or .xlsx")

    pefts = df.columns[2:]

    if columns_to_skip is None:
        columns_to_skip = []

    # if os.path.exists(factscore_results_file):
    #     fs_df = pd.read_csv(factscore_results_file)
    # else:
    #     # fs_df = pd.DataFrame(columns=["peft_name", "score", "num_atomic_facts"])
    #     fs_df = pd.DataFrame(columns=["test_domain_dataset", "peft_name", "score", "num_atomic_facts"])
    if not os.path.exists(factscore_results_file):
        fs_df = pd.DataFrame(columns=["test_domain_dataset", "peft_name", "score", "num_atomic_facts"])
        fs_df.to_csv(factscore_results_file, index=False)

    print("Data Directory: ", STORE_DATA_DIR)

    fs_reslts = []

    for _peft in pefts:
        # skipping zero shot for now.
        try: 
            # if _peft in columns_to_skip:
            if _peft in columns_to_skip:
                continue
            print("Running FactScore for: Domain: {} - Dataset: {} - Peft: {}".format(domain, dataset_name, _peft))
            new_df = df[["article", "truth", _peft]]  # summary
            prediction_col_name = "data-{}_{}-peft-{}".format(domain, dataset_name, _peft)

            # reading the file to check for values if they exist:
            print("Checking for existing data - peft values")
            fs_df = pd.read_csv(factscore_results_file)
            if ((fs_df["test_domain_dataset"] == "{}_{}".format(domain, dataset_name)) & (
                    fs_df["peft_name"] == _peft)).any():
                # ("{}_{}".format(domain, dataset_name) in fs_df["test_domain_dataset"].values
                # and _peft in fs_df["peft_name"].values):
                print("!!!! Skipping already calculated FactScore for: Domain: {} - Dataset: {} - Peft: {} !!!!".format(
                    domain, dataset_name, _peft))
                continue
            fs_df = None

            new_df.rename(columns={_peft: prediction_col_name}, inplace=True)

            jsonl_path = df_to_jsonl_for_factscore(df=new_df, predictions_col_name=prediction_col_name,
                                                main_data_dir=STORE_DATA_DIR)

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
            fs_df = pd.read_csv(factscore_results_file)
            if ((fs_df["test_domain_dataset"] == res_obj["test_domain_dataset"]) & (
                    fs_df["peft_name"] == res_obj["peft_name"])).any():
                if "test_domain_dataset" not in fs_df.columns:
                    fs_df["test_domain_dataset"] = None  # Add with default value if missing

                # Ensure the column exists in obj
                if "test_domain_dataset" in res_obj:
                    fs_df.loc[
                        (fs_df["peft_name"] == res_obj["peft_name"]) & (
                                    fs_df["test_domain_dataset"] == res_obj["test_domain_dataset"]),
                        list(res_obj.keys())
                    ] = list(res_obj.values())
                    fs_df.to_csv(factscore_results_file, index=False)
                else:
                    print("'test_domain_dataset' is missing in obj")
            else:
                # Append the new row
                fs_df = pd.concat([fs_df, pd.DataFrame([res_obj])], ignore_index=True)
                fs_df.to_csv(factscore_results_file, index=False)
            print("Writtent Obj: \n{}".format(res_obj))
        
        except Exception as e:
            print("Exception caught in {}: {} ".format(_peft, e))

        # fs_reslts.append(res_obj)

    # for obj in fs_reslts:
    #     if ((fs_df["test_domain_dataset"] == obj["test_domain_dataset"]) & (fs_df["peft_name"] == obj["peft_name"])).any():
    #     # if obj["test_domain_dataset"] in fs_df["test_domain_dataset"].values and obj["peft_name"] in fs_df["peft_name"].values:
    #         # Update the row where the "Dataset" value matches
    #         # fs_df.loc[fs_df["peft_name"] == obj["peft_name"], obj.keys()] = obj.values()
    #         # fs_df.loc[fs_df["peft_name"] == obj["peft_name"] and fs_df["test_domain_data"] == obj["test_domain_data"], obj.keys()] = obj.values()

    #         # fs_df.loc[
    #         #     (fs_df["peft_name"] == obj["peft_name"]) & (fs_df["test_domain_data"] == obj["test_domain_data"]),
    #         #     list(obj.keys())
    #         # ] = list(obj.values())
    #         if "test_domain_dataset" not in fs_df.columns:
    #             fs_df["test_domain_dataset"] = None  # Add with default value if missing

    #         # Ensure the column exists in obj
    #         if "test_domain_dataset" in obj:
    #             fs_df.loc[
    #                 (fs_df["peft_name"] == obj["peft_name"]) & (fs_df["test_domain_dataset"] == obj["test_domain_dataset"]),
    #                 list(obj.keys())
    #             ] = list(obj.values())
    #         else:
    #             print("'test_domain_dataset' is missing in obj")

    #     else:
    #         # Append the new row
    #         fs_df = pd.concat([fs_df, pd.DataFrame([obj])], ignore_index=True)

    # fs_df.to_csv(factscore_results_file, index=False)


if __name__ == "__main__":
    # Please install the requirements.txt, have a .env file with required tokens( OPENAI, huggingface), and api.key file with openai token
    model_name: str = "GPT-4o-mini"
    grounding_provided: bool = True
    openai_key = "api.key"

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")
    parser.add_argument("--data_dir", default="", type=str, help="Path to main data directory")
    parser.add_argument("--domain", default=None, type=str, help="Domain for which to generate FActScores")
    # parser.add_argument("--best_summaries_yaml", default=None, type=str, help="Domain for which to generate FActScores")

    main_args = parser.parse_args()
    STORE_DATA_DIR = main_args.data_dir
    _domain = main_args.domain

    # SUMMARIES for HUMAN EVAL

    # human_eval_files = glob.glob("summaries/human_eval/human_eval_summaries_*.*")

    # for f_name in human_eval_files:
    #     fs_results_file = "summaries/human_eval/human_eval_factscore_results_unseen_test_25samples.csv"
    #     dataset_name = f_name.split("_25samples.")[0].split("_")[-1]
    #     print("Calculating Factscore for Dataset Name: {} - {}".format(dataset_name, f_name))
    #     generating_factscores_for_summaries(model_name, grounding_provided, openai_key, "unseen_data", dataset_name,
    #                                         factscore_results_file=fs_results_file, summary_file_path=f_name)

    hvd = ["news", "legal", "scientific", "medical"] if _domain is None else [_domain] # - rerun

    for dd in hvd:
        fs_results_file = "summaries/paper/factscore_results_hvd_{}.csv".format(dd)
        # dataset_name = f_name.split("_25samples.")[0].split("_")[-1]
        f_name = "summaries/summaries_unseen_test_{}.{}".format(dd, "xlsx" if dd in ["medical", "legal"] else "csv")
        print("Calculating Factscore for Dataset Name: {} - {}".format(dd, f_name))
        generating_factscores_for_summaries(model_name, grounding_provided, openai_key, "unseen_data", dd,
                                            factscore_results_file=fs_results_file, summary_file_path=f_name)

    # TODO: READING SUMMARIES Files from a YAML
    # best_summaries_yaml = main_args.best_summaries_yaml

    # if best_summaries_yaml is not None:
    #     list_of_best_summaries = read_yaml(best_summaries_yaml)
    #     """
    #     files:
    #       scientific: "best_scientific_summaries_unseen_data_files_list.csv"
    #       medical: "best_medical_summaries_unseen_data_files_list.xlsx"
    #       legal: "best_legal_summaries_unseen_data_files_list.xlsx"
    #       news:  "best_news_summaries_unseen_data_files_list.csv"
    #     """
    #     for dataset_name, file_name in list_of_best_summaries["files"].items():
    #         fs_results_file = "summaries/best_summaries_factscore_results.csv".format()
    #         generating_factscores_for_summaries(model_name, grounding_provided, openai_key, "unseen_data", dataset_name,
    #                                             factscore_results_file=fs_results_file, summary_file_path=file_name)
    #
    # else:
    #     # domain = "scientific"
    #     # dataset_name = "arxiv"
    #     from dataset_lib import datasets_info_dict, SumDomains
    #     datasets_dict = {}

    # did = datasets_info_dict.pop(SumDomains.UNSEEN_TEST)
    # if _domain is None:
    #     raise ValueError("Please provide a domain to calculate FactScore for, your options are: {} or pass 'all'".format(list(datasets_info_dict.keys())))
    # if _domain == "all":
    #     datasets_info_dict.pop(SumDomains.UNSEEN_TEST)
    #     datasets_dict = datasets_info_dict
    # else:
    #     datasets_dict[SumDomains(_domain)] = datasets_info_dict[SumDomains(_domain)]
    #
    # for domain, datasets in datasets_dict.items():
    #     for dataset_name in datasets:
    #         print("Calculating FactScore for: Domain: {} - Dataset: {}".format(domain.name.lower(), dataset_name))
    #         if not "unseen_test" == domain.name.lower():
    #             f_name = "summaries/summaries_{}_{}_150samples.csv".format(domain.name.lower(), dataset_name)
    #             fs_results_file = factscore_results_file
    #         else:
    #             f_name = "summaries/summaries_{}_{}_25samples.csv".format(domain.name.lower(), dataset_name)
    #             fs_results_file = "summaries/factscore_results_unseen_test_25samples.csv"
    #         generating_factscores_for_summaries(model_name, grounding_provided, openai_key, domain.name.lower(), dataset_name,
    #                                             factscore_results_file=fs_results_file, summary_file_path=f_name)

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
