import os
import random
from enum import Enum

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

# from .domains_utils import datasets_info_dict, SumDatasets, preprocessing_scientific_or_medical, \
#     preprocessing_legal, preprocessing_news, preprocessing_low_resource_domain

DATASET_STORAGE_DIR = ""   # fetch from configs

DEFAULT_SYSTEM_PROMPT = """
    You are an AI assistant that excels at summarizing long-form articles. Please provide a concise and informative summary of the following article provided by the user.
""".strip()
# Given below is an article. Write a concise and informative Summary for the article.

class SumDatasets(Enum):
    ARXIV = "scientific"  # , "scientific"
    PUBMED = "medical"  # , "medical"
    MULTI_LEX = "legal"
    NEWS = "news"


datasets_info_dict = {
    SumDatasets.ARXIV.name: {
        "dataset_id": "ccdv/arxiv-summarization",
        "local_path": "domains/arxiv_summarization",
        "source": "hugging_face"
    },
    SumDatasets.PUBMED.name: {
        "dataset_id": "ccdv/pubmed-summarization",
        "local_path": "domains/pubmed_summarization",
        "source": "hugging_face"
    },
    SumDatasets.MULTI_LEX.name: {
        "dataset_id": "allenai/multi_lexsum",
        "local_path": "domains/legal_summarization",
        "source": "hugging_face"
    },
    SumDatasets.NEWS.name: {
        "dataset_id": "",
        "local_path": "domains/news_summarization",
        "source": "csv_file"
    }
}


def generate_training_prompt(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    prompt = """### Instruction: {}\n\n### Article: {}\n\n### Summary: {}""".format(system_prompt, article, summary)
    return prompt.strip()


def llama3_training_prompt(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    # prompt = """
    #     <|begin_of_text|><|start_of_message|>system<|end_of_message|>\n{}\n
    #     <|start_of_message|>user<|end_of_message|>\nArticle:\n{}<|end_of_message|>\n
    #     <|start_of_message|>assistant<|end_of_message|>\n{}<|end_of_text|>
    # """.format(system_prompt, article, summary)

    # This done after reading from the chat_template using tokenize=False
    prompt = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>\n
            <|start_header_id|>user<|end_header_id|>\nPlease provide the summary for the article:\n{}<|eot_id|>\n
            <|start_header_id|>assistant<|end_header_id|>\nHere is your Summary:\n{}<|eot_id|><|end_of_text|>
        """.format(system_prompt, article, summary)

    return prompt.strip()


def llama3_testing_prompt(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    # prompt = """
    #     <|begin_of_text|><|start_of_message|>system<|end_of_message|>\n{}\n
    #     <|start_of_message|>user<|end_of_message|>\nArticle:\n{}<|end_of_message|>\n
    #     <|start_of_message|>assistant<|end_of_message|>\n
    # """.format(system_prompt, article)

    # This done after reading from the chat_template using tokenize=False
    prompt = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>\n
            <|start_header_id|>user<|end_header_id|>\nPlease provide the summary for the article:\n{}<|eot_id|>\n
            <|start_header_id|>assistant<|end_header_id|>\nHere is your Summary:\n
            """.format(system_prompt, article)

    return prompt.strip()


def chat_template_prompt_training(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "Article:\n{}".format(article)},
        {"role": "assistant", "content": "SUMMARY:\n{}".format(summary)}
    ]
    return messages


def chat_template_prompt_inference(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "Article:\n{}".format(article)}
    ]

    return messages


def inference_prompt(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    prompt = """### Instruction: {}\n### Article: {}\n### Summary:""".format(system_prompt.strip(), article.strip())

    return prompt.strip()


def preprocessing_data_with_chat_format(sample):
    chats = [chat_template_prompt_training(article=article, summary=summary)
             # generate_training_prompt(article=article, summary=summary)
             for article, summary in zip(sample["article"], sample["abstract"])]
    sample.pop("article", None)
    sample.pop("abstract", None)
    return {
        "chat": chats
    }


def preprocessing_scientific_or_medical(sample):
    texts = [llama3_training_prompt(article=article, summary=summary)
        # generate_training_prompt(article=article, summary=summary)
             for article, summary in zip(sample["article"], sample["abstract"])]
    sample.pop("article", None)
    sample.pop("abstract", None)
    return {
        "text": texts
    }
    # Process Scientific or Medical
    # Process Legal dataset
    # Process News


def preprocessing_legal(sample):
    pass
# exp = list(dataset["validation"])[4]
    #
    # print(exp["sources"])
    #
    # for sum_len in ["long", "short", "tiny"]:
    #     print(exp["summary/" + sum_len])  # Summaries of three lengths


def preprocessing_news(sample):
    pass


def preprocessing_low_resource_domain(sample):
    pass


def read_news_summarization():
    file_path = "domains/news_summarization.csv"

    df = pd.read_csv(file_path)

    # import pdb; pdb.set_trace()


class SumDataLoader:

    def __init__(self, dataset_name: str, force_download: bool = False, training_samples: int = 1000,
                 eval_samples: int = 1000, test_samples: int = 500, src_directory: str = "", chat_template: bool = False,
                 preprocess_function: callable = None, sort_dataset_on_article_len: bool = False):

        self.src_directory = src_directory
        self.chat_template = chat_template
        self.dataset = SumDatasets(dataset_name.lower())
        self.preprocess_function = preprocess_function
        self.sort_dataset_on_article_len = sort_dataset_on_article_len
        self.dataset_name = self.dataset.name

        self.force_download = force_download
        self.training_samples = training_samples
        self.eval_samples = eval_samples
        self.test_samples = test_samples

        self.dataset_id = datasets_info_dict[self.dataset_name]["dataset_id"]
        self.local_path = datasets_info_dict[self.dataset_name]["local_path"]
        self.dataset_source = datasets_info_dict[self.dataset_name]["source"]

        self.train_set, self.validation_set, self.test_set = None, None, None

    def print_dataset_stats(self):
        print("**Dataset Stats**\nTrain: {}\n\nValidation: {}\n\nTest: {}\n\n".format(self.train_set,
                                                                                      self.validation_set,
                                                                                      self.test_set))

    @classmethod
    def sample_dataset(cls, dataset_split, sample_size=1000):
        random.seed(42)
        sampled_data = []
        # Manually iterate through the IterableDataset
        for idx, example in enumerate(dataset_split):
            # Decide whether to include the example based on the current index
            if len(sampled_data) < sample_size:
                # Always add samples if we have not reached the desired sample size
                sampled_data.append(example)
            else:
                # Once we have enough samples, replace elements with decreasing probability
                j = random.randint(0, idx)
                if j < sample_size:
                    sampled_data[j] = example

        return sampled_data
        # sampled_indices = random.sample(range(dataset_split.num_rows), sample_size)
        # return dataset_split.select(sampled_indices)

    def loading_dataset_from_hf(self):
        local_path = self.src_directory + self.local_path
        if not os.path.exists(local_path) or self.force_download:
            if self.dataset_name == SumDatasets.MULTI_LEX.name:
                loaded_dataset = load_dataset(path=self.dataset_id, name='v20230518', streaming=True, trust_remote_code=True)
            else:
                loaded_dataset = load_dataset(path=self.dataset_id, streaming=True, trust_remote_code=True)
            # dataset_dict = {}
            # for split, data in dataset.items():
            #     # if split == "train":
            #     data = list(data)[:500]  # TODO: selecting random 1k or 5k or 10k
            #     df = pd.DataFrame(data)
            #     dataset_dict[split] = Dataset.from_pandas(df)  # dataset[split].to_pandas())
            # _dataset = DatasetDict(dataset_dict)

            self.train_set = Dataset.from_list(self.sample_dataset(loaded_dataset["train"], self.training_samples))
            self.validation_set = Dataset.from_list(self.sample_dataset(loaded_dataset["validation"]))
            self.test_set = Dataset.from_list(self.sample_dataset(loaded_dataset["test"]))

            dataset_dict = DatasetDict({
                "train": self.train_set,
                "validation": self.validation_set,
                "test": self.test_set
            })

            dataset_dict.save_to_disk(local_path)
            print("******* Dataset for Domain - '{}' is stored at {} !!!*******".format(self.dataset_name, local_path))
            # del dataset_dict
            # return _dataset
        else:
            loaded_dataset = load_from_disk(local_path)

            if self.dataset_name in [SumDatasets.ARXIV.name, SumDatasets.PUBMED.name]:
                loaded_dataset = loaded_dataset.map(lambda x: {"article_tokens": len(x["article"].split()),
                                                               "abstract_tokens": len(x["abstract"].split())})

                loaded_dataset = loaded_dataset.filter(lambda x: (x["article_tokens"] != 0 and
                                                                  x["abstract_tokens"] != 0))

                print("Min and Max number of tokens in articles in train dataset of {} are {} and {} "
                      "respectively!".format(self.dataset_name, min(loaded_dataset["train"]["article_tokens"]),
                                             max(loaded_dataset["train"]["article_tokens"])))

                if self.sort_dataset_on_article_len:
                    print("Sorting test data on number of tokens of articles")
                    loaded_dataset = loaded_dataset.sort("article_tokens")
                loaded_dataset = loaded_dataset.remove_columns(["article_tokens", "abstract_tokens"])

                self.train_set = loaded_dataset["train"].select(range(self.training_samples))
                self.validation_set = loaded_dataset["validation"].select(range(self.eval_samples))
                self.test_set = loaded_dataset["test"].select(range(self.test_samples))
                del loaded_dataset

        self.print_dataset_stats()

        # return self.train_set, self.validation_set, self.test_set

    def loading_dataset_from_csv_or_excel(self):
        # TODO: To be implemented
        pass
        raise NotImplementedError()
        # return None, None, None

    def loading_dataset_splits(self):
        if self.dataset_source == "hugging_face":
            return self.loading_dataset_from_hf()

        elif self.dataset_source in ["csv_file", "excel_file"]:
            return self.loading_dataset_from_csv_or_excel()

        else:
            NotImplementedError("~~~Sources except hugging_face and tabular (csv or excel) NOT IMPLEMENTED YET~~~")

    # def __get_preprocess_function(self):
    #     if self.dataset_name in [SumDatasets.ARXIV.name, SumDatasets.PUBMED.name]:
    #         return preprocessing_scientific_or_medical
    #
    #     elif self.dataset_name == SumDatasets.MULTI_LEX:
    #         return preprocessing_legal
    #
    #     elif self.dataset_name == SumDatasets.NEWS:
    #         return preprocessing_news
    #
    #     else:
    #         return preprocessing_low_resource_domain

    def processing_data_with_training_prompt(self, dataset_split: Dataset, preprocess_function: callable = None):
        if preprocess_function is not None:
            self.preprocess_function = preprocess_function
        return (dataset_split.map(self.preprocess_function, batched=True
                                  ).shuffle(seed=42))

        # dataset_split = dataset_split.map(preprocess_function, batched=True
        #                           ).shuffle(seed=42)
        #
        # dataset_split.to_csv("splitsssss.csv", index=False)
        # return dataset_split

    def processing_data_with_test_prompt(self, dataset_split: Dataset, preprocess_function: callable = None):
        if preprocess_function is not None:
            self.preprocess_function = preprocess_function
        return (dataset_split.map(self.preprocess_function, batched=True
                                  ).shuffle(seed=42))

    def tokenization_of_data_splits(self, tokenization_process: callable):
        # cols_to_remove = [col for col in self.train_set.column_names if col != "input_ids"]
        cols_to_remove = [col for col in self.train_set.column_names if col not in ["input_ids", "attention_mask", "labels"]]
        print("cols to remove: ", cols_to_remove)
        # print("texxxxttt: ", self.train_set[0].keys())
        self.train_set = self.train_set.map(tokenization_process, batched=True, remove_columns=cols_to_remove)
        # self.train_set = self.train_set.remove_columns(
        #     [col for col in self.train_set.column_names if col != "input_ids"])

        self.validation_set = self.validation_set.map(tokenization_process, batched=True, remove_columns=cols_to_remove)
        # self.validation_set = self.validation_set.remove_columns(
        #     [col for col in self.validation_set.column_names if col != "input_ids"])

        return self.train_set, self.validation_set, self.test_set
