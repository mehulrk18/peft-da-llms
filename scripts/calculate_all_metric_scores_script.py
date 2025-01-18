import glob
import os
import statistics

import pandas as pd
from dotenv import load_dotenv
from utils.evaluation_metrics_llms import meteor_metric, rouge_metric, bertscore_metric, bleu_metric

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


def calculate_metrics_for_all_columns_in_file(summary_file, score_file, column_to_leave=None):
    # if domain == "unseen_test":
    #     df = pd.read_csv("summaries/summaries_{}_{}.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    # else:
    #     df = pd.read_csv("summaries/summaries_{}_{}_150samples.csv".format(domain, dataset_name), encoding="ISO-8859-1")
    if column_to_leave is None:
        column_to_leave = []
    df = pd.read_csv(summary_file)
    pefts = df.columns[2:]
    metrics = {"rouge": ["model", "rouge1", "rouge2", "rougeL", "rougeLsum"],
               "bertscore": ["model", "precision_mean", "precision_median", "recall_mean", "recall_median", "f1_mean",
                             "f1_median"],
               "meteor": ["model", "meteor"],
               "bleu": ["model", "bleu", "precisions", "brevity_penalty", "length_ratio", "translation_length",
                        "reference_length"]}
    rouge = rouge_metric()
    bertscore = bertscore_metric()
    meteor = meteor_metric()
    bleu = bleu_metric()

    for _metric in metrics.keys():
        if not os.path.exists(score_file.format(_metric)):
            m_df = pd.DataFrame(columns=metrics[_metric])
            m_df.to_csv(score_file.format(_metric), index=False)

    print("Data Directory: ", STORE_DATA_DIR)

    for _peft in pefts:
        # skipping zero shot for now.
        metric_scores = {k: {} for k in metrics.keys()}
        if _peft in column_to_leave:
            continue
        print("Calculating Meteor for summaries from file: {} Peft: {}".format(summary_file, _peft))
        if "unseen_test" not in summary_file:
            model_name = summary_file.split("summaries_")[1].split("_50samples")[0] + "_" + _peft
            model_name = model_ckpt.get(model_name, model_name)
        else:
            if "zero_shot_instruct" == _peft:
                _domain = summary_file.split("summaries_unseen_test_")[1].split("_25samples.csv")[0]
                model_name = "zero_shot_unseen_test_{}_results".format(_domain)
            else:
                model_name = _peft

        references = df["truth"].tolist()
        predictions = df[_peft].tolist()

        r_score = rouge.compute(predictions=predictions, references=references)
        m_score = meteor.compute(predictions=predictions, references=references)
        bl_score = bleu.compute(predictions=predictions, references=references)

        _bs_score = bertscore.compute(predictions=predictions, references=references, lang="en", verbose=True)
        bs_score = {}
        bs_score["precision"] = {"mean": statistics.mean(_bs_score["precision"]),
                                 "median": statistics.median(_bs_score["precision"])}
        bs_score["recall"] = {"mean": statistics.mean(_bs_score["recall"]),
                              "median": statistics.median(_bs_score["recall"])}
        bs_score["f1"] = {"mean": statistics.mean(_bs_score["f1"]),
                          "median": statistics.median(_bs_score["f1"])}

        metric_scores["rouge"] = {
            "model": model_name,
            "rouge1": r_score["rouge1"],
            "rouge2": r_score["rouge2"],
            "rougeL": r_score["rougeL"],
            "rougeLsum": r_score["rougeLsum"]
        }
        metric_scores["bleu"] = {
            "model": model_name,
            "bleu": bl_score["bleu"],
            "precisions": bl_score["precisions"],
            "brevity_penalty": bl_score["brevity_penalty"],
            "length_ratio": bl_score["length_ratio"],
            "translation_length": bl_score["translation_length"],
            "reference_length": bl_score["reference_length"]
        }

        metric_scores["meteor"] = {
            "model": model_name,
            "meteor": m_score["meteor"]
        }

        metric_scores["bertscore"] = {
            "model": model_name,
            "precision_mean": bs_score["precision"]["mean"],
            "precision_median": bs_score["precision"]["median"],
            "recall_mean": bs_score["recall"]["mean"],
            "recall_median": bs_score["recall"]["median"],
            "f1_mean": bs_score["f1"]["mean"],
            "f1_median": bs_score["f1"]["median"]
        }
        import json
        print("Scores for file: {} and peft {} are \n {}\n".format(summary_file, _peft, json.dumps(metric_scores, indent=4)))

        for _metric in metrics.keys():
            f_name = score_file.format(_metric)
            m_df = pd.read_csv(f_name)
            # Append the new row
            m_df = pd.concat([m_df, pd.DataFrame([metric_scores[_metric]])], ignore_index=True)
            m_df.to_csv(f_name, index=False)
            print("Written {} Scores to: {}".format(_metric, f_name))


if __name__ == "__main__":
    files = glob.glob("summaries/summaries_*_50samples.csv")
    files.extend(glob.glob("summaries/summaries_unseen_test_*_25samples.csv"))
    for file in files:
        print("Calculating All metrics for: ", file)
        if "unseen_test" in file:
            score_file = "summaries/unseen_data_{}_scores25.csv"
            calculate_metrics_for_all_columns_in_file(summary_file=file, score_file=score_file)
        else:
            score_file = "summaries/{}_scores50.csv"
            calculate_metrics_for_all_columns_in_file(summary_file=file, score_file=score_file,
                                                      column_to_leave=["zero_shot_instruct"])