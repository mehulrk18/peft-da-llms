import glob
import os

import pandas as pd
from dotenv import load_dotenv
from utils.evaluation_metrics_llms import meteor_metric

load_dotenv()

factscore_results_file = "summaries/factscore_results.csv"
STORE_DATA_DIR = ""

model_ckpt = {'zero_shot_legal_multilex_results': 'zero_shot_legal_multilex_results',
              'zero_shot_legal_eurlex_results': 'zero_shot_legal_eurlex_results',
              'zero_shot_legal_billsum_results': 'zero_shot_legal_billsum_results',
              'zero_shot_medical_pubmed_results': 'zero_shot_medical_pubmed_results',
              'zero_shot_medical_cord19_results': 'zero_shot_medical_cord19_results',
              'zero_shot_medical_scilay_results': 'zero_shot_medical_scilay_results',
              'zero_shot_news_cnndm_results': 'zero_shot_news_cnndm_results',
              'zero_shot_news_multinews_results': 'zero_shot_news_multinews_results',
              'zero_shot_news_xsum_results': 'zero_shot_news_xsum_results',
              'zero_shot_news_newsroom_results': 'zero_shot_news_newsroom_results',
              'zero_shot_scientific_arxiv_results': 'zero_shot_scientific_arxiv_results',
              'zero_shot_scientific_elsevier_results': 'zero_shot_scientific_elsevier_results',
              'zero_shot_scientific_scitldr_results': 'zero_shot_scientific_scitldr_results',
              'legal_billsum_adalora': 'legal_billsum_adalora_checkpoint-1250',
              'legal_billsum_ia3': 'legal_billsum_ia3_checkpoint-1000',
              'legal_billsum_loha': 'legal_billsum_loha_checkpoint-750',
              'legal_billsum_lokr': 'legal_billsum_lokr_checkpoint-1000',
              'legal_billsum_lora': 'legal_billsum_lora_checkpoint-1250',
              'legal_billsum_oft': 'legal_billsum_oft_checkpoint-1000',
              'legal_eurlex_adalora': 'legal_eurlex_adalora_checkpoint-3250',
              'legal_eurlex_ia3': 'legal_eurlex_ia3_checkpoint-1300',
              'legal_eurlex_loha': 'legal_eurlex_loha_checkpoint-3250',
              'legal_eurlex_lokr': 'legal_eurlex_lokr_checkpoint-3250',
              'legal_eurlex_lora': 'legal_eurlex_lora_checkpoint-3250',
              'legal_eurlex_oft': 'legal_eurlex_oft_checkpoint-3250',
              'legal_multilex_adalora': 'legal_multilex_adalora_checkpoint-4075',
              'legal_multilex_ia3': 'legal_multilex_ia3_checkpoint-4075',
              'legal_multilex_loha': 'legal_multilex_loha_checkpoint-4075',
              'legal_multilex_lokr': 'legal_multilex_lokr_checkpoint-4075',
              'legal_multilex_lora': 'legal_multilex_lora_checkpoint-4075',
              'legal_multilex_oft': 'legal_multilex_oft_checkpoint-3260',
              'scientific_arxiv_loha': 'scientific_arxiv_loha_checkpoint-1480',
              'scientific_arxiv_lokr': 'scientific_arxiv_lokr_checkpoint-1480',
              'zero_shot_medical_mslr_results': 'zero_shot_medical_mslr_results',
              'scientific_arxiv_lora': 'scientific_arxiv_lora_checkpoint-592',
              'scientific_arxiv_oft': 'scientific_arxiv_oft_checkpoint-1184',
              'scientific_elsevier_adalora': 'scientific_elsevier_adalora_checkpoint-1255',
              'scientific_elsevier_ia3': 'scientific_elsevier_ia3_checkpoint-502',
              'scientific_elsevier_loha': 'scientific_elsevier_loha_checkpoint-753',
              'scientific_elsevier_lokr': 'scientific_elsevier_lokr_checkpoint-753',
              'scientific_elsevier_lora': 'scientific_elsevier_lora_checkpoint-1255',
              'scientific_elsevier_oft': 'scientific_elsevier_oft_checkpoint-1255',
              'scientific_scitldr_adalora': 'scientific_scitldr_adalora_checkpoint-1250',
              'scientific_scitldr_ia3': 'scientific_scitldr_ia3_checkpoint-1250',
              'scientific_scitldr_loha': 'scientific_scitldr_loha_checkpoint-1000',
              'scientific_scitldr_lokr': 'scientific_scitldr_lokr_checkpoint-1000',
              'scientific_scitldr_lora': 'scientific_scitldr_lora_checkpoint-1250',
              'scientific_scitldr_oft': 'scientific_scitldr_oft_checkpoint-1250',
              'medical_cord19_adalora': 'medical_cord19_adalora_checkpoint-1330',
              'medical_cord19_ia3': 'medical_cord19_ia3_checkpoint-1064',
              'medical_cord19_loha': 'medical_cord19_loha_checkpoint-1064',
              'medical_cord19_lokr': 'medical_cord19_lokr_checkpoint-1064',
              'medical_cord19_lora': 'medical_cord19_lora_checkpoint-1330',
              'medical_cord19_oft': 'medical_cord19_oft_checkpoint-1064',
              'medical_pubmed_adalora': 'medical_pubmed_adalora_checkpoint-1250',
              'medical_pubmed_ia3': 'medical_pubmed_ia3_checkpoint-250',
              'medical_pubmed_loha': 'medical_pubmed_loha_checkpoint-1250',
              'medical_pubmed_lokr': 'medical_pubmed_lokr_checkpoint-1250',
              'medical_pubmed_lora': 'medical_pubmed_lora_checkpoint-1250',
              'medical_pubmed_oft': 'medical_pubmed_oft_checkpoint-1250',
              'medical_scilay_adalora': 'medical_scilay_adalora_checkpoint-2140',
              'medical_scilay_ia3': 'medical_scilay_ia3_checkpoint-1284',
              'medical_scilay_loha': 'medical_scilay_loha_checkpoint-1284',
              'medical_scilay_lokr': 'medical_scilay_lokr_checkpoint-1284',
              'medical_scilay_lora': 'medical_scilay_lora_checkpoint-2140',
              'medical_scilay_oft': 'medical_scilay_oft_checkpoint-2140',
              'medical_mslr_adalora': 'medical_mslr_adalora_checkpoint-1250',
              'medical_mslr_ia3': 'medical_mslr_ia3_checkpoint-1250',
              'medical_mslr_loha': 'medical_mslr_loha_checkpoint-1000',
              'medical_mslr_lokr': 'medical_mslr_lokr_checkpoint-1000',
              'medical_mslr_lora': 'medical_mslr_lora_checkpoint-1250',
              'medical_mslr_oft': 'medical_mslr_oft_checkpoint-1250',
              'news_cnndm_adalora': 'news_cnndm_adalora_checkpoint-1250',
              'news_cnndm_ia3': 'news_cnndm_ia3_checkpoint-500',
              'news_cnndm_loha': 'news_cnndm_loha_checkpoint-1000',
              'news_cnndm_lokr': 'news_cnndm_lokr_checkpoint-1000',
              'news_cnndm_lora': 'news_cnndm_lora_checkpoint-1250',
              'news_cnndm_oft': 'news_cnndm_oft_checkpoint-1250',
              'news_multinews_adalora': 'news_multinews_adalora_checkpoint-1250',
              'news_multinews_ia3': 'news_multinews_ia3_checkpoint-250',
              'news_multinews_loha': 'news_multinews_loha_checkpoint-1000',
              'news_multinews_lokr': 'news_multinews_lokr_checkpoint-1000',
              'news_multinews_lora': 'news_multinews_lora_checkpoint-1250',
              'news_newsroom_adalora': 'news_newsroom_adalora_checkpoint-1000',
              'news_multinews_oft': 'news_multinews_oft_checkpoint-1250',
              'news_newsroom_ia3': 'news_newsroom_ia3_checkpoint-1000',
              'news_newsroom_loha': 'news_newsroom_loha_checkpoint-1250',
              'news_newsroom_lokr': 'news_newsroom_lokr_checkpoint-1250',
              'news_newsroom_lora': 'news_newsroom_lora_checkpoint-1000',
              'news_xsum_adalora': 'news_xsum_adalora_checkpoint-1000',
              'news_xsum_ia3': 'news_xsum_ia3_checkpoint-250',
              'news_xsum_loha': 'news_xsum_loha_checkpoint-1000',
              'news_xsum_lokr': 'news_xsum_lokr_checkpoint-1000',
              'scientific_arxiv_adalora': 'scientific_arxiv_adalora_checkpoint-1480',
              'scientific_arxiv_ia3': 'scientific_arxiv_ia3_checkpoint-592',
              'news_xsum_lora': 'news_xsum_lora_checkpoint-1000',
              'news_newsroom_oft': 'news_newsroom_oft_checkpoint-1250',
              'news_xsum_oft': 'news_xsum_oft_checkpoint-1250'}


def calculate_meteor_for_all_columns_in_file(summary_file, meteor_score_file, column_to_leave=None):
    # if domain == "unseen_test":
    #     df = pd.read_csv("summaries/summaries_{}_{}.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    # else:
    #     df = pd.read_csv("summaries/summaries_{}_{}_150samples.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    if column_to_leave is None:
        column_to_leave = []
    df = pd.read_csv(summary_file)
    pefts = df.columns[2:]

    meteor = meteor_metric()

    if not os.path.exists(meteor_score_file):
        m_df = pd.DataFrame(columns=["model", "meteor"])
        m_df.to_csv(meteor_score_file, index=False)

    print("Data Directory: ", STORE_DATA_DIR)

    fs_reslts = []

# if "unseen_test" not in summary_file:
    for _peft in pefts:
        # skipping zero shot for now.
        if _peft in column_to_leave:
            continue
        print("Calculating Meteor for summaries from file: {} Peft: {}".format(summary_file, _peft))
        if "unseen_test" not in summary_file:
            model_name = summary_file.split("summaries_")[1].split("_150samples")[0] + "_" + _peft
            model_name = model_ckpt.get(model_name, model_name)
        else:
            if "zero_shot_instruct" == _peft:
                _domain = summary_file.split("summaries_unseen_test_")[1].split(".csv")[0]
                model_name = "zero_shot_unseen_test_{}_results".format(_domain)
            else:
                model_name = _peft

        references = df["truth"].tolist()
        predictions = df[_peft].tolist()

        # reading the file to check for values if they exist:
        print("Checking for existing model - meteor values")
        m_df = pd.read_csv(meteor_score_file)
        if model_name in m_df["model"].values:
            # ("{}_{}".format(domain, dataset_name) in fs_df["test_domain_dataset"].values
            # and _peft in fs_df["peft_name"].values):
            print("!!!! Skipping already calculated Meteor for: model: {} - Peft: {} !!!!".format(
                model_name,  _peft))
            continue

        m_score = meteor.compute(predictions=predictions, references=references)
        meteor_scores = {
            "model": model_name,
            "meteor": m_score["meteor"]
        }

        print("Meteor Scores: {}".format(meteor_scores), "\n\n")

        m_df = pd.read_csv(meteor_score_file)
        # Append the new row
        m_df = pd.concat([m_df, pd.DataFrame([meteor_scores])], ignore_index=True)
        m_df.to_csv(meteor_score_file, index=False)
        print("Written Object")


if __name__ == "__main__":
    files = glob.glob("summaries/summaries_*150samples.csv")
    files.extend(glob.glob("summaries/summaries_unseen_test_*.csv"))
    for file in files:
        print("Calculating Meteor for: ", file)
        if "unseen_test" in file:
            meteor_score_file = "summaries/unseen_data_meteor_scores.csv"
            calculate_meteor_for_all_columns_in_file(summary_file=file, meteor_score_file=meteor_score_file)
        else:
            meteor_score_file = "summaries/meteor_scores.csv"
            calculate_meteor_for_all_columns_in_file(summary_file=file, meteor_score_file=meteor_score_file,
                                                     column_to_leave=["zero_shot_instruct"])