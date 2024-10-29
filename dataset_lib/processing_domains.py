import os
import random
from enum import Enum

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

# from .domains_utils import datasets_info_dict, SumDatasets, preprocessing_scientific_or_medical, \
#     preprocessing_legal, preprocessing_news, preprocessing_low_resource_domain

DOMAIN = None
DATASET_STORAGE_DIR = ""   # fetch from configs

# DEFAULT_SYSTEM_PROMPT = """
#     You are an AI assistant that excels at summarizing long-form articles. Please provide a concise and informative summary of the following article provided by the user.
# """.strip()

DEFAULT_SYSTEM_PROMPT = """
    Summarize the given article by addressing the following key points:

    1. Main Topic: What is the primary subject or theme of the article?

    2. Context: What background information is provided to frame the discussion?

    3. Key Arguments: What are the main arguments or points made by the author?

    4. Evidence and Examples: What supporting evidence or examples are presented?

    5. Conclusions: What conclusions does the author draw from the discussion?

    6. Implications: What are the potential implications or significance of the article's content for its audience?
""".strip()

SAMPLE_PROMPT = """ 
    Summarize the given article by including the following key points:

    1. Objective: What is the main research question or objective of the study?

    2. Background: What is the context or rationale for the study?

    3. Methods: What study design, population, and methodologies were used?

    4. Key Findings: What are the most significant results or discoveries from the study?

    5. Conclusions: What conclusions do the authors draw from their findings?

    6. Clinical Relevance: How might the study’s findings impact medical practice or patient care?
    """


class SumDatasets(Enum):
    ARXIV = "scientific"
    PUBMED = "medical"
    MULTI_LEX = "legal"
    NEWS = "news"


datasets_info_dict = {
    SumDatasets.ARXIV.name: {
        "dataset_id": "ccdv/arxiv-summarization",
        "local_path": "domains/arxiv_summarization",
        "version": None,
        "columns_to_remove": [],
        "source": "hugging_face"
    },
    SumDatasets.PUBMED.name: {
        "dataset_id": "ccdv/pubmed-summarization",
        "local_path": "domains/pubmed_summarization",
        "version": None,
        "columns_to_remove": [],
        "source": "hugging_face"
    },
    SumDatasets.MULTI_LEX.name: {
        "dataset_id": "allenai/multi_lexsum",
        "local_path": "domains/multi_lex_summarization",
        "version": "v20230518",
        "columns_to_remove": ["id", "sources", "summary/long", "summary/short", "summary/tiny", "case_metadata", "sources_metadata"],
        "source": "hugging_face"
    },
    SumDatasets.NEWS.name: {
        "dataset_id": "",
        "local_path": "domains/news_summarization",
        "version": None,
        "columns_to_remove": [],
        "source": "csv_file"
    }
}


DEFAULT_DOMAIN_PROMPT = {
    SumDatasets.ARXIV.name: """
        Provide a summary of the given scientific article that includes the following elements:

        1. Objective: What is the main research question or hypothesis?

        2. Background: What is the theoretical context or motivation for the research?

        3. Methods: Detail the approach, experiments, or simulations conducted.

        4. Key Findings: What are the principal results or contributions of the paper?

        5. Conclusions: What insights or conclusions do the authors derive from their research?

        6. Broader Impact: How does this research contribute to its field or influence future work?""".strip(),

    SumDatasets.PUBMED.name: """
        Summarize the given medical study by addressing the following key points:

        1. Objective: What is the primary research question or aim of the study?

        2. Background: What is the clinical context or rationale behind the research?

        3. Methods: Describe the study design, population involved, and methodologies employed.

        4. Key Findings: Highlight the most important results or discoveries of the study.

        5. Conclusions: What conclusions do the authors reach based on their findings?

        6. Clinical Implications: How might these findings influence clinical practice or patient outcomes?""".strip(),

    SumDatasets.MULTI_LEX.name: """
        Summarize the given legal case study by focusing on the following aspects:

        1. Case Background: What is the context and significance of the case?

        2. Legal Question: What are the primary legal issues or questions being addressed?

        3. Arguments: What are the main arguments presented by both sides?

        4. Rulings: What decisions were made by the court, and on what basis?

        5. Key Precedents: Are there important precedents cited that influence the case?

        6. Implications: What are the potential implications of the ruling on future cases or legal interpretations?
        """.strip(),

    SumDatasets.NEWS.name: """
        Summarize the given news article by capturing the following key points:

        1. Main Event: What is the primary event or issue being reported?

        2. Context: What background information is necessary to understand the significance of the news?

        3. Key Details: What are the most critical facts or figures related to the story?

        4. Reactions: How have different stakeholders or the public responded to the event?

        5. Implications: What are the potential consequences or future developments related to this news?

        6. Closing Statement: What is the overarching message or takeaway from the article?""".strip()
}


def llama3_training_prompt(article: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    # This done after reading from the chat_template using tokenize=False
    prompt = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>\n
            <|start_header_id|>user<|end_header_id|>\nPlease provide the summary for the article:\n{}<|eot_id|>\n
            <|start_header_id|>assistant<|end_header_id|>\nHere is your Summary:\n{}<|eot_id|><|end_of_text|>
        """.format(system_prompt, article, summary)

    return prompt.strip()


def llama3_testing_prompt(article: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
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


def preprocessing_data_with_chat_format(sample):
    chats = [chat_template_prompt_training(article=article, summary=summary)
             for article, summary in zip(sample["article"], sample["abstract"])]
    sample.pop("article", None)
    sample.pop("abstract", None)
    return {
        "chat": chats
    }


def preprocessing_data_with_prompt(sample):
    texts = [llama3_training_prompt(article=article, summary=summary, system_prompt=DEFAULT_DOMAIN_PROMPT[DOMAIN])
             for article, summary in zip(sample["article"], sample["abstract"])]
    sample.pop("article", None)
    sample.pop("abstract", None)
    return {
        "text": texts
    }


def preprocessing_news(sample):
    pass


def preprocessing_low_resource_domain(sample):
    pass


def read_news_summarization():
    file_path = "domains/news_summarization.csv"

    df = pd.read_csv(file_path)


class SumDataLoader:

    def __init__(self, dataset_name: str, force_download: bool = False, training_samples: int = 5000,
                 eval_samples: int = 1000, test_samples: int = 1000, src_directory: str = "", chat_template: bool = False,
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
        self.dataset_version = datasets_info_dict[self.dataset_name]["version"]
        self.columns_to_remove_in_preprocessing = datasets_info_dict[self.dataset_name]["columns_to_remove"]

        self.train_set, self.validation_set, self.test_set = None, None, None
        global DOMAIN
        DOMAIN = self.dataset_name

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

                j = random.randint(0, idx)
                if j < sample_size:
                    sampled_data[j] = example

        return sampled_data

    def loading_dataset_from_hf(self):
        local_path = self.src_directory + self.local_path
        if not os.path.exists(local_path) or self.force_download:
            if self.dataset_name == SumDatasets.MULTI_LEX.name:
                loaded_dataset = load_dataset(path=self.dataset_id, name=self.dataset_version, streaming=True,
                                              trust_remote_code=True)
                def preprocess_multi_lex_data_after_downloading(sample):
                    article = ["\n".join(text) for text in sample["sources"]]
                    summary = [summ for summ in sample["summary/long"]]
                    sample.pop("sources", None)
                    sample.pop("summary/long", None)
                    return {
                        "article": article,
                        "abstract": summary
                    }
                loaded_dataset = loaded_dataset.map(preprocess_multi_lex_data_after_downloading, batched=True)
                loaded_dataset = loaded_dataset.remove_columns(self.columns_to_remove_in_preprocessing)
            else:
                loaded_dataset = load_dataset(path=self.dataset_id, streaming=True, trust_remote_code=True)

            self.train_set = Dataset.from_list(self.sample_dataset(loaded_dataset["train"], self.training_samples))
            self.validation_set = Dataset.from_list(self.sample_dataset(loaded_dataset["validation"], self.eval_samples))
            self.test_set = Dataset.from_list(self.sample_dataset(loaded_dataset["test"], self.test_samples))

            loaded_dataset = DatasetDict({
                "train": self.train_set,
                "validation": self.validation_set,
                "test": self.test_set
            })

            loaded_dataset.save_to_disk(local_path)
            print("******* Dataset for Domain - '{}' is stored at {} !!!*******".format(self.dataset_name, local_path))
        else:
            loaded_dataset = load_from_disk(local_path)

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
        self.train_set = loaded_dataset["train"].select(range(min(self.training_samples, len(loaded_dataset["train"]))))
        self.validation_set = loaded_dataset["validation"].select(range(min(self.eval_samples, len(loaded_dataset["validation"]))))
        self.test_set = loaded_dataset["test"].select(range(min(self.test_samples, len(loaded_dataset["test"]))))
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

    def processing_data_with_training_prompt(self, dataset_split: Dataset, preprocess_function: callable = None):
        if preprocess_function is not None:
            self.preprocess_function = preprocess_function
        return (dataset_split.map(self.preprocess_function, batched=True
                                  ).shuffle(seed=42))

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