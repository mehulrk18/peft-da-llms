# Imports
import os.path
import re
import json
import time
import random
from collections import Counter
from functools import lru_cache

import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
import nltk

from ml_helpers import clear_cache, run_regression, normalize_features, derive_baseline_features

nltk.download('wordnet')
from local_datasets import LoadDatasetFromLocal, LocalDatasetArxiv, LocalDatasetElsevier, LocalDatasetPubmed, \
    LocalDatasetCord19, LocalDatasetMSLR, LocalDatasetSciLay, LocalDatasetSciTLDR, LocalDatasetBillSum, \
    LocalDatasetEurLex, LocalDatasetMultiLex, LocalDatasetCNNDailyMail, LocalDatasetNewsRoom, LocalDatasetMultiNews, \
    LocalDatasetXSumNews, LocalDatasetScientific, LocalDatasetMedical, LocalDatasetLegal, LocalDatasetNews
from domains import DomainsDataset
from analysing_data import Similarity
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
# from ds.supported import load_dataset
from typing import List, Union, Sequence, Mapping
import pandas as pd
from datetime import datetime
import random
import logging
import itertools

# Definitions


translate_dataset_name = {
    "scientific": {
        "arxiv": LocalDatasetArxiv,
        "elsevier": LocalDatasetElsevier,
        "scitldr": LocalDatasetSciTLDR,
    },
    "medical": {
        "cord19": LocalDatasetCord19,
        "mslr": LocalDatasetMSLR,
        "pubmed": LocalDatasetPubmed,
        "scilay": LocalDatasetSciLay,
    },
    "legal": {
        "billsum": LocalDatasetBillSum,
        "eurlex": LocalDatasetEurLex,
        "multilex": LocalDatasetMultiLex
    },
    "news": {
        "cnndm": LocalDatasetCNNDailyMail,
        "multinews": LocalDatasetMultiNews,
        "newsroom": LocalDatasetNewsRoom,
        "xsum": LocalDatasetXSumNews,
    },
    "unseen_test": {
        "scientific": LocalDatasetScientific,
        "medical": LocalDatasetMedical,
        "legal": LocalDatasetLegal,
        "news": LocalDatasetNews,
    }
}


def find_domain(dataset_name):
    for domain in translate_dataset_name.keys():
        if dataset_name in translate_dataset_name[domain]:
            return domain


def load_domain_dataset(
        domain_name: str,
        dataset_name: str,
        preview: bool = False,
        samples: Union[int, str] = "max",
        min_input_size: int = 0,
        data_files: Union[
            dict, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None
        ] = None,
) -> LoadDatasetFromLocal:
    logging.info(f"Preparing dataset {dataset_name} from domain {domain_name}")

    assert domain_name in list(translate_dataset_name.keys())
    assert dataset_name in list(translate_dataset_name[domain_name].keys())

    return translate_dataset_name[domain_name][dataset_name](
        preview=preview,
        samples=samples,
        min_input_size=min_input_size,
        # load_csv=load_csv,
        # data_files=data_files,
    )


def get_task_spec_metrics(domain: str, task: str, task_spec_metrics):
    task_spec_metrics = task_spec_metrics.drop(["ds", "split", "model"], axis=1)
    if task == "classification":
        return {"accuracy": random.uniform(0, 1)}
    elif task == "summarization":
        if len(task_spec_metrics) > 0:
            score_dict = task_spec_metrics.iloc[0].to_dict()
        else:  # return random values
            logging.warning(f"Added random value for task: {task}, domain:{domain} \n")
            score_dict = dict()
            for score in task_spec_metrics.columns:
                score_dict[score] = random.uniform(0, 1)
        # @todo: drop the non-relavant features here so they are not included in y_weighted.
        return score_dict
    else:
        return {}


def get_domain_specific_metrics(domain: str, ds_name: str, num_samples=100):
    d = get_domain_dataset(domain=domain, ds_name=ds_name, split="test", num_samples=num_samples)
    return {"learning_difficult": d.compute_learning_difficulty()}


def get_ttl_hash(seconds=86400):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


@lru_cache(maxsize=128)
def get_domain_dataset(domain, ds_name, split, num_samples=5, ttl_hash=None):
    dataset = load_domain_dataset(
        domain_name=domain,
        dataset_name=ds_name,
        samples=num_samples,
    )

    articles = dataset.get_split(split)["text"]
    if len(articles) > num_samples:
        articles = articles[:num_samples]

    d = DomainsDataset(articles, domain)
    return d


def get_domain_similarity_metrics(src_domain: str, source: str, tgt_domain: str, target: str, da: str, num_samples=100):
    # s_dataset = load_domain_dataset(
    #     domain_name=src_domain,
    #     dataset_name=source,
    #     samples=num_samples,
    # )
    # if source == target and da == "in-domain-adapt":
    #
    #     s_dval = s_dataset.get_split("train")['text']
    # else:
    #     s_dval = s_dataset.get_split("test")['text']
    #     s_dval = s_dval[:num_samples]
    #
    # t_dataset = load_domain_dataset(
    #     domain_name=tgt_domain,
    #     dataset_name=target,
    #     samples=num_samples,
    # )
    # t_dval = t_dataset.get_split("test")['text']
    # t_dval = t_dval[:num_samples]
    #
    # # client = OpenAI()
    # client = None
    #
    # S = DomainsDataset(file_paths=s_dval, file_names=source, client=client)
    # T = DomainsDataset(file_paths=t_dval, file_names=target, client=client)  # , unique=True)
    # ST = Similarity(S, T)
    #
    # return {"word-overlap": ST.word_overlap,
    #         "vocab-overlap": ST.vocab_overlap,
    #         "relevance-overlap": ST.relevance_overlap,
    #         # "renyi-divergence": ST.renyi_divergence,
    #         "kl-divergence": ST.kl_divergence,
    #         # "js-divergence": ST.js_divergence,
    #         }
    if source == target and da == "in-domain-adapt":
        source_split = "train"
        target_split = "test"
    elif source == target and da == "no-domain-adapt":
        source_split = "train"
        target_split = "train"
    else:
        source_split = "test"
        target_split = "test"

        # start = time.time()
    S = get_domain_dataset(domain=src_domain, ds_name=source, split=source_split, num_samples=num_samples, ttl_hash=get_ttl_hash())
    # end = time.time()
    # print(f"{source} domain computation took {end - start}")

    # start = time.time()
    T = get_domain_dataset(domain=tgt_domain, ds_name=target, split=target_split, num_samples=num_samples, ttl_hash=get_ttl_hash())
    # end = time.time()
    # print(f"{source} domain computation took {end - start}")

    ST = Similarity(S, T, ttl_hash=get_ttl_hash())

    return {
        "vocab-overlap": ST.vocab_overlap,
        "tf-idf-overlap": ST.tf_idf_overlap,
        "source_shannon_entropy": S.shannon_entropy,
        "target_shannon_entropy": T.shannon_entropy,
        "kl-divergence": ST.kl_divergence,
        "contextual-overlap": ST.contextual_overlap,
        # "js-divergence": ST.js_divergence,
    }


def weighted_average(nums, weights):
    return sum(x * y for x, y in zip(nums, weights)) / sum(weights)


# def get_features(da: str, src_domain, source: str, tgt_domain, target: str, task: str, task_scores) -> (List, List):
#     features = []
#     feature_names = ['da-type', 'source', 'target', ]
#     features.append(da)
#     features.append(source)
#     features.append(target)
#
#     domain_spec_features = get_domain_specific_metrics(target)
#     features += list(domain_spec_features.values())
#     feature_names += list(domain_spec_features.keys())
#
#     domain_similarity_features = get_domain_similarity_metrics(tgt_domain, target, src_domain, source, da,
#                                                                num_samples=100)
#     features += list(domain_similarity_features.values())
#     feature_names += list(domain_similarity_features.keys())
#
#     try:
#         if source == target and da == "in-domain-adapt":
#             source_task_scores = task_scores.loc[
#                 (task_scores['dataset_name'] == source) & (task_scores['split'] == 'train')]
#         else:
#             source_task_scores = task_scores.loc[
#                 (task_scores['dataset_name'] == source) & (task_scores['split'] == 'test')]
#     except:
#         # todo: add a dummy variable for this
#         print(f"Failed to get scores for domain {source} for {da} setting. Assigning dummy scores.")
#
#     task_specific_feature = get_task_spec_metrics(source, task, source_task_scores)
#     features += list(task_specific_feature.values())
#     feature_names += [f'source_{key}' for key in list(task_specific_feature.keys())]
#     feature_weight = [1 / len(task_specific_feature.values())] * len(
#         task_specific_feature.values())  # equal weight to all features
#     weighted_y_source = weighted_average(list(task_specific_feature.values()), feature_weight)
#
#     target_task_scores = task_scores.loc[task_scores['dataset_name'] == target]
#     task_specific_feature = get_task_spec_metrics(target, task, target_task_scores)
#     features += list(task_specific_feature.values())
#     feature_names += [f'target_{key}' for key in list(task_specific_feature.keys())]
#     feature_weight = [1 / len(task_specific_feature.values())] * len(
#         task_specific_feature.values())  # equal weight to all features
#     weighted_y_target = weighted_average(list(task_specific_feature.values()), feature_weight)
#
#     y_drop = weighted_y_source - weighted_y_target
#
#     features += [weighted_y_target, weighted_y_source, y_drop]
#     feature_names += ['y_weighted_source', 'y_weighted_target', 'y_drop']
#
#     return features, feature_names


# def get_features(da: str, src_domain, source: str, tgt_domain, target: str, task: str, task_scores) -> (List, List):
def get_features(
    da: str, src_domain: str, source: str, tgt_domain:str, target: str, num_samples, ft=False,  task: str = "",
        task_scores=None) -> (List, List):
    features = []
    feature_names = [
        "da-type",
        "source",
        "target",
        "ft"
    ]
    features.append(da)
    features.append(source)
    features.append(target)
    features.append(ft)

    domain_spec_features = get_domain_specific_metrics(domain=tgt_domain, ds_name=target, num_samples=num_samples)
    features += list(domain_spec_features.values())
    feature_names += list(domain_spec_features.keys())

    domain_similarity_features = get_domain_similarity_metrics(
        source=source, src_domain=src_domain, target=target, tgt_domain=source, da=da, num_samples=num_samples
    )
    features += list(domain_similarity_features.values())
    feature_names += list(domain_similarity_features.keys())

    # try:
    #     source_model = "Meta-Llama-3-8B-Instruct"
    #     target_model = "Meta-Llama-3-8B-Instruct"
    #     # source_model = "meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo"
    #     # target_model = "meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo"
    #     if source == target and da == "in-domain-adapt":
    #         target_split = "test"
    #         source_split = "test"
    #         # if ft:
    #         #     source_model = "meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo"
    #         #     target_model = "anumafzal94-llama3.1"
    #
    #     elif source == target and da == "no-domain-adapt":
    #         target_split = "test"
    #         source_split = "train"
    #         # if ft:
    #         #     source_model = "anumafzal94-llama3.1"
    #         #     target_model = "anumafzal94-llama3.1"
    #         #     source_split = "test"
    #
    #     else:
    #         target_split = "test"
    #         source_split = "test"
    #         # if ft:
    #         #     source_model = "anumafzal94-llama3.1"
    #         #     target_model = "anumafzal94-llama3.1"
    #
    #     source_task_scores = task_scores.loc[
    #         (task_scores["ds"] == source) & (task_scores["split"] == source_split) & (task_scores["model"] == source_model)
    #     ]
    # except:
    #     # todo: add a dummy variable for this
    #     print(
    #         f"Failed to get scores for domain {source} for {da} setting. Assigning dummy scores."
    #     )

    # task_specific_feature = get_task_spec_metrics(source, task, source_task_scores)
    # features += list(task_specific_feature.values())
    # feature_names += [f"source_{key}" for key in list(task_specific_feature.keys())]
    # feature_weight = [1 / len(task_specific_feature.values())] * len(task_specific_feature.values())  # equal weight to all features
    # weighted_y_source = weighted_average(
    #     list(task_specific_feature.values()), feature_weight
    # )

    # target_task_scores = task_scores.loc[(task_scores["ds"] == target) & (task_scores["split"] == target_split) & (task_scores["model"] == target_model)]
    # task_specific_feature = get_task_spec_metrics(target, task, target_task_scores)
    # features += list(task_specific_feature.values())
    # feature_names += [f"target_{key}" for key in list(task_specific_feature.keys())]
    # feature_weight = [1 / len(task_specific_feature.values())] * len(task_specific_feature.values())  # equal weight to all features
    # weighted_y_target = weighted_average(
    #     list(task_specific_feature.values()), feature_weight
    # )

    # y_drop = weighted_y_source - weighted_y_target

    # features += [weighted_y_target, weighted_y_source, y_drop]
    # feature_names += ["y_weighted_source", "y_weighted_target", "y_drop"]

    return features, feature_names

# def get_template(scores_path:str, domains, ) -> pd.DataFrame:
#
#     task_scores = pd.read_excel(scores_path,header=0)
#     task_scores = task_scores.drop(['run_id', 'model', 'prompt'], axis = 1)
#     domains = list(set(task_scores["dataset_name"]))
#     #domains.remove("samsum")
#     task_scores = task_scores.drop(columns=task_scores.columns[:19])
#     da_type = ["in-domain-adapt", "single-domain-adapt", "no-domain-adapt"] #, "multi-domain-adapt"]
#     task = 'summarization'
#     feature_names = ['dummy_feature_name'] * (len(task_scores.columns) - 1)
#     df = pd.DataFrame()
#
#     for da in tqdm(da_type):
#         features = []
#         features.append(da)
#         if da == "in-domain-adapt" or da == "no-domain-adapt":
#             for domain in tqdm(domains):
#                 features, feature_names = get_features(da,dataset_domain_mapping[domain], domain,
#                                                        dataset_domain_mapping[domain], domain, task, task_scores)
#                 if df.columns.empty:
#                     df = pd.DataFrame(columns=feature_names)
#                 df.loc[len(df)] = features
#
#         elif da == "single-domain-adapt":
#             for source in tqdm(domains):
#                 domains_copy = domains.copy()
#                 domains_copy.remove(source)
#                 for target in domains_copy:
#                     features, feature_names = get_features(da, dataset_domain_mapping[source], source,
#                                                            dataset_domain_mapping[target], target, task, task_scores)
#                     if df.columns.empty:
#                         df = pd.DataFrame(columns=feature_names)
#                     df.loc[len(df)] = features
#         else:
#             df.loc[len(df)] = [numpy.NaN for i in range(len(feature_names))]
#
#     write_logs(df)
#     return df

def get_template(task_scores: pd.DataFrame, num_datasets=None, num_samples=10, ft=False) -> pd.DataFrame:
    try:
        domain_datasets = list(set(task_scores["ds"]))
    except Exception as e:
        print("except: ", e)
        domain_datasets = [str(v) for _, v in translate_dataset_name.items() for k in v.keys()]
    # domains = ['arxiv', 'gigaword', 'wispermed', 'govreport']
    if num_datasets is not None:
        if len(domain_datasets) > num_datasets:
            domain_datasets = domain_datasets[:num_datasets]
    # task_scores = task_scores.drop(columns=task_scores.columns[:19])
    da_type = ["in-domain-adapt", "single-domain-adapt", "no-domain-adapt"]
    task = "summarization"
    feature_names = ["dummy_feature_name"] * (len(task_scores.columns) - 1)
    df = pd.DataFrame()

    for da in tqdm(da_type):
        features = []
        features.append(da)
        if da == "in-domain-adapt" or da == "no-domain-adapt":
            for _dataset in tqdm(domain_datasets):
                _domain = find_domain(_dataset)
                features, feature_names = get_features(
                    da=da, source=_dataset, src_domain=_domain, tgt_domain=_domain, target=_dataset, task=task,
                    task_scores=task_scores, num_samples=num_samples, ft=ft
                )
                if df.columns.empty:
                    df = pd.DataFrame(columns=feature_names)
                df.loc[len(df)] = features

        elif da == "single-domain-adapt":
            for source in tqdm(domain_datasets):
                domain_datasets_copy = domain_datasets.copy()
                domain_datasets_copy.remove(source)
                src_domain = find_domain(source)
                for target in domain_datasets_copy:
                    tgt_domain = find_domain(target)
                    features, feature_names = get_features(
                        da=da, source=source, src_domain=src_domain, target=target, tgt_domain=tgt_domain, task=task,
                        task_scores=task_scores, num_samples=num_samples, ft=ft
                    )
                    if df.columns.empty:
                        df = pd.DataFrame(columns=feature_names)
                    df.loc[len(df)] = features
        else:
            df.loc[len(df)] = [np.NaN for i in range(len(feature_names))]
    # clear_cache()
    write_logs(df)

    return df


def write_logs(df: pd.DataFrame):
    date_time = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.now())
    folder = os.path.join("logs", date_time)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "features.csv"), index=False)

    print(f"Run logs would be locally stored at {folder}")


# def construct_training_corpus(domains: List, da_type: str = "in-domain-adapt",
#                               template_path: str = "template.xlsx") -> pd.DataFrame:
#
#     assert da_type in ["in-domain-adapt", "single-domain-adapt", "multi-domain-adapt"]
#
#     template = get_template(template_path, domains = domains)
#     print (template)
#     return template


def construct_training_corpus(
        num_datasets: int,
        num_samples,
        da_type: str = "in-domain-adapt",
        template_path: str = "template.xlsx",
) -> pd.DataFrame:
    assert da_type in ["in-domain-adapt", "single-domain-adapt"]

    df = pd.read_excel(template_path, header=0)
    try:
        df_zero_shot = df.loc[df['model'] == 'Meta-Llama-3-8B-Instruct']
    # df_zero_shot = df.loc[df['model'] == 'meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo']
        df_ft = df.loc[(df['split'] == 'test') & (df['ds'] != 'aclsum')]
    except Exception as e:
        print(f"Error: {e}")
        df_zero_shot = pd.DataFrame([])
        df_ft = pd.DataFrame([])



    template_2 = get_template(df_ft, num_datasets=num_datasets, num_samples=num_samples, ft=True
                              )
    template_2.to_excel("template2_ft.xlsx")
    template_1 = get_template(df_zero_shot, num_datasets=num_datasets, num_samples=num_samples
                              )
    template_1.to_excel("template1.xlsx")

    # print (template)
    template = pd.concat([template_1, template_2], axis=0)
    return template


# if __name__ == '__main__':
#     load_dotenv()
#
#     get_domain_similarity_metrics()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--da_type',
    #                     type=str,
    #                     default="in-domain-adapt")
    # parser.add_argument('--domain',
    #                     dest="domains",
    #                     action='append',
    #                     default = ['arxiv', 'pubmed', 'govreport', 'wispermed'])
    #
    # parser.add_argument('--template_path',
    #                     type=str,
    #                     default="template.xlsx")
    #
    # args = parser.parse_args()
    # construct_training_corpus(domains=args.domains, da_type=args.da_type,template_path=args.template_path)


if __name__ == '__main__':
    load_dotenv("/dss/dsshome1/03/ge85giy2/domainadaptationllms.env")

    parser = argparse.ArgumentParser()
    parser.add_argument('--da_type',
                        type=str,
                        default="in-domain-adapt")
    parser.add_argument('--dataset',
                        dest="datasets",
                        action='append',
                        default=["arxiv", "elsevier", "scitldr", "cord19", "mslr", "pubmed", "scilay", "billsum",
                                 "eurlex", "multilex", "cnndm", "multinews", "newsroom", "xsum", "scientific", "medical",
                                 "legal", "news"]
    )
                        # default=['arxiv', 'pubmed', 'govreport', 'wispermed', 'cnndm', 'samsum', 'bigpatent',
                        #          'billsum', ])

    parser.add_argument('--template_path',
                        type=str,
                        default= "template.xlsx") # "inference_results/inference_results_ds_13_500_all.xlsx")

    args = parser.parse_args()
    num_samples = 500
    total_datasets = 18 #14
    minumum_datasets = 3
    cache = True
    sklearn_feature_selection = [True, False]
    selected_feat_rouge = []
    selected_feat_all = []


    all_scores = None
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    directory = f"training_features/{date_time}"
    cache_directory = "training_features/2024-11-14_11-39-15"
    if cache:
        directory = cache_directory
    # file_name= "training_features_ds_13_llama3.1_8b_0-shot_samples500.xlsx"
    file_name = "training_features_ds_13_llama3_8b_0-shot_samples500.xlsx"
    file_name = os.path.join(directory, file_name)
    if cache and os.path.isfile(file_name):
        all_features = pd.read_excel(file_name)
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
        all_features = construct_training_corpus(num_datasets=total_datasets, da_type=args.da_type,
                                                 template_path=args.template_path, num_samples=num_samples)

        all_features.to_excel(file_name)

    #all_features = all_features.loc[all_features['ft']==False]
    for feat_selection in sklearn_feature_selection:
        for n in range(minumum_datasets, total_datasets + 1, 1):
            print(f"Number of domains: {n}")



            domains_to_compute = (list(all_features['source'].unique()))[:n]
            features = all_features.loc[all_features['source'].isin(domains_to_compute)]
            # 1) Prepare Baseline Features

            # 1.1) Raw features
            features_baseline = derive_baseline_features(features)
            #print (f"Baseline Features: {features_baseline.columns}")
            #scores_baseline_raw , selected_feat= run_regression(features_baseline, mode='baseline-raw')

            # 1.2) Normalized features -> check if even needed
            features_baseline_norm = normalize_features(features_baseline)
            scores_baseline_norm, selected_feat = run_regression(features_baseline_norm, mode='baseline-norm', feature_selection_bool=feat_selection)
            selected_feat_rouge.extend(selected_feat )

            # 1.3) Normalized and reduced features
            #features_baseline_norm_red = derive_baseline_features_red(features_baseline_norm)
            #scores_baseline_norm_red , selected_feat= run_regression(features_baseline_norm_red, mode='baseline-norm-red')

            # 2) Prepare normal features

            # 2.1) Raw Features
            #features = pd.read_excel(file_name)
            #print(f"All Features: {features.columns}")
            #scores_all_raw, selected_feat = run_regression(features, mode='all-raw')

            # 2.2) Normalized Features
            features_norm = normalize_features(features)
            scores_all_norm  , selected_feat= run_regression(features_norm, mode='all-norm', feature_selection_bool=feat_selection)
            selected_feat_all.extend(selected_feat)

            # 2.3) Reduced Feature Space
            #features_norm_reduced = reduce_features_all(features_norm)
            #scores_all_norm_red = run_regression(features_norm_reduced, mode='all-red', feature_selection_bool=feat_selection)


            pd_scores = pd.DataFrame.from_records([scores_baseline_norm, scores_all_norm])
            pd_scores['num_datasets'] = [n] * len(pd_scores)
            #print (pd_scores)
            file_name = f"scores_ds_{n}_llama3.1_8b_0-shot_{num_samples}.xlsx"
            file_name = os.path.join(directory, file_name)
            pd_scores.to_excel(file_name)
            if all_scores is None:
                all_scores = pd_scores
                #print (pd_scores.columns)
            else:
                all_scores = pd.concat([all_scores, pd_scores], axis=0)

        # file_name = f"scores_llama3.1_8b_0-shot_{num_samples}.xlsx"
        file_name = f"scores_llama3_8b_0-shot_{num_samples}.xlsx"
        file_name = os.path.join(directory, file_name)
        all_scores.to_excel(file_name)
        print (f"final scores stored at: {file_name}")
        print(f"Feature Selection Baseline : {Counter(selected_feat_rouge)}")
        print (f"Feature Selection : {Counter(selected_feat_all)}")
clear_cache()