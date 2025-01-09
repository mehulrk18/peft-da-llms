import os.path

import pandas as pd
from dotenv import load_dotenv

from FActScore.fact_score import calc_factscore
from utils.fscore_utils import df_to_jsonl_for_factscore

load_dotenv()


factscore_results_file = "summaries/factscore_results.csv"

def generating_factscores_for_summaries(model_name, grounding_provided, open_ai_key, domain, dataset_name):
    if domain == "unseen_test":
        df = pd.read_csv("summaries/summaries_{}_{}.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    else:
        df = pd.read_csv("summaries/summaries_{}_{}_150samples.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    pefts = df.columns[2:]

    if os.path.exists(factscore_results_file):
        fs_df = pd.read_csv(factscore_results_file)
    else:
        fs_df = pd.DataFrame(columns=["peft_name", "score", "num_atomic_facts"])

    fs_reslts = []

    for peft in pefts:
        print("Running FactScore for: Domain: {} - Dataset: {} - Peft: {}".format(domain, dataset_name, peft))
        new_df = df[["article", "truth", peft]]  # summary
        prediction_col_name = "{}_{}_{}".format(domain, dataset_name, peft)

        new_df.rename(columns={peft: prediction_col_name}, inplace=True)

        jsonl_path = df_to_jsonl_for_factscore(df=new_df, predictions_col_name=prediction_col_name)

        arg_dict = {
            "input_path": jsonl_path,
            "model_name": model_name,
            "grounding_provided": grounding_provided,
            "openai_key": open_ai_key,
        }

        result_scores = calc_factscore(arg_dict=arg_dict)
        print("FACTSCORE for {}: {}".format(prediction_col_name, result_scores), "\n\n")

        res_obj = {
            "peft_name": prediction_col_name,
            "score": result_scores["score"],
            "num_atomic_facts": result_scores["num_atomic_facts"]
        }
        fs_reslts.append(res_obj)

    for obj in fs_reslts:
        if obj["peft_name"] in fs_df["peft_name"].values:
            # Update the row where the "Dataset" value matches
            fs_df.loc[fs_df["peft_name"] == obj["peft_name"], obj.keys()] = obj.values()
        else:
            # Append the new row
            fs_df = pd.concat([fs_df, pd.DataFrame([obj])], ignore_index=True)

    fs_df.to_csv(factscore_results_file, index=False)


if __name__ == "__main__":
    # Please install the requirements.txt, have a .env file with required tokens( OPENAI, huggingface), and api.key file with openai token
    model_name: str = "GPT-4o-mini"
    grounding_provided: bool = True
    openai_key = "api.key"

    domain = "scientific"
    dataset_name = "arxiv"
    generating_factscores_for_summaries(model_name, grounding_provided, openai_key, domain, dataset_name)

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
