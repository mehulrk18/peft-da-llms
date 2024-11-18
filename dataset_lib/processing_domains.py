import os
import random

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

from .constants import *


def llama3_training_prompt(content: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    # This done after reading from the chat_template using tokenize=False
    prompt = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>\n
            <|start_header_id|>user<|end_header_id|>\nPlease provide the summary for the article:\n{}<|eot_id|>\n
            <|start_header_id|>assistant<|end_header_id|>\nHere is your Summary:\n{}<|eot_id|><|end_of_text|>
        """.format(system_prompt, content, summary)

    return prompt.strip()


def llama3_testing_prompt(content: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    # This done after reading from the chat_template using tokenize=False
    prompt = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>\n
            <|start_header_id|>user<|end_header_id|>\nPlease provide the summary for the article:\n{}<|eot_id|>\n
            <|start_header_id|>assistant<|end_header_id|>\nHere is your Summary:\n
            """.format(system_prompt, content)

    return prompt.strip()


def chat_template_prompt_training(content: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "Article:\n{}".format(content)},
        {"role": "assistant", "content": "SUMMARY:\n{}".format(summary)}
    ]
    return messages


def chat_template_prompt_inference(content: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "Article:\n{}".format(content)}
    ]

    return messages


def preprocessing_data_with_chat_format(sample):
    chats = [chat_template_prompt_training(content=content, summary=summary)
             for content, summary in zip(sample["content"], sample["summary"])]
    sample.pop("content", None)
    sample.pop("summary", None)
    return {
        "chat": chats
    }


def preprocessing_data_with_prompt(sample):
    texts = [llama3_training_prompt(content=content, summary=summary, system_prompt=DEFAULT_SYSTEM_PROMPT)
             for content, summary in zip(sample["content"], sample["summary"])]
    sample.pop("content", None)
    sample.pop("summary", None)
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

    def __init__(self, domain: str, dataset_name: str, force_download: bool = False, training_samples: int = 5000,
                 eval_samples: int = 1000, test_samples: int = 1000, src_directory: str = "",
                 chat_template: bool = False,
                 preprocess_function: callable = None, sort_dataset_on_article_len: bool = False):

        self.src_directory = src_directory
        self.chat_template = chat_template
        self.domain = SumDomains(domain.lower())
        self.dataset_name = dataset_name
        self.preprocess_function = preprocess_function
        self.sort_dataset_on_article_len = sort_dataset_on_article_len

        self.force_download = force_download
        self.training_samples = training_samples
        self.eval_samples = eval_samples
        self.test_samples = test_samples

        self.dataset_info = datasets_info_dict[self.domain][self.dataset_name]
        # self.dataset_id = datasets_info_dict[self.domain][self.dataset_name]["dataset_id"]
        # self.local_path = datasets_info_dict[self.domain][self.dataset_name]["local_path"]
        # self.dataset_source = datasets_info_dict[self.domain][self.dataset_name]["source"]
        # self.dataset_version = datasets_info_dict[self.domain][self.dataset_name]["version"]
        # self.columns_to_remove_in_preprocessing = datasets_info_dict[self.domain][self.dataset_name][
        #     "columns_to_remove"]

        self.train_set, self.validation_set, self.test_set = None, None, None

    def __str__(self):
        return "DataLoader for Domain: {} and Dataset: {}\n{}".format(self.domain, self.dataset_name,
                                                                      self.return_stats())

    def return_stats(self):
        stat = "**Dataset Stats**\nTrain: {}\n\nValidation: {}\n\nTest: {}\n\n".format(self.train_set,
                                                                                       self.validation_set,
                                                                                       self.test_set)
        return stat

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
        local_path = self.src_directory + self.dataset_info.local_path
        if not os.path.exists(local_path) or self.force_download:
            if self.dataset_name in ["arxiv", "pubmed"]:
                def process_data_after_downloading(sample):
                    content = [text for text in sample["article"]]
                    summary = [summ for summ in sample["abstract"]]
                    return {
                        "content": content,
                        "summary": summary
                    }
                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_data_after_downloading, batched=True)

            elif self.dataset_name == "multi_lex":
                def process_multi_lex_data_after_downloading(sample):
                    content = ["\n".join(text) for text in sample["sources"]]
                    summary = [summ for summ in sample["summary/long"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_multi_lex_data_after_downloading, batched=True)

            elif self.dataset_name == "cnn_dm":
                def process_cnn_dm_data_after_downloading(sample):
                    content = [text for text in sample["article"]]
                    summary = [summ for summ in sample["highlights"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_cnn_dm_data_after_downloading, batched=True)

            elif self.dataset_name == "multi_news":
                def process_multi_news_data_after_downloading(sample):
                    content = [text for text in sample["document"]]
                    summary = [summ for summ in sample["summary"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_multi_news_data_after_downloading, batched=True)

            elif self.dataset_name == "x_sum":
                def process_x_sum_data_after_downloading(sample):
                    content = [text for text in sample["document"]]
                    summary = [summ for summ in sample["summary"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_x_sum_data_after_downloading, batched=True)

            elif self.dataset_name == "newsroom":
                def process_newsroom_data_after_downloading(sample):
                    content = [text for text in sample["text"]]
                    summary = [summ for summ in sample["summary"]]
                    return {
                        "content": content,
                        "summary": summary
                    }
                extract_location = "domains/news/{}".format(self.dataset_info.dataset_id)
                try:
                    import tarfile
                    with tarfile.open("domains/news/newsroom-release.tar", "r") as tar:
                        tar.extractall(extract_location)
                except Exception as e:
                    print("Exception caught: ", e)
                    print("Downloading the newsroom dataset from the URL: {}\n"
                          "And store it at path domains/news/newsroom-release.tar\n **EXITING**".format(self.dataset_info["download_url"]))
                    exit(1)
                loaded_dataset = load_dataset(path=extract_location, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_newsroom_data_after_downloading, batched=True)

            elif self.dataset_name == "elsevier":
                def process_elsevier_data_after_downloading(sample):
                    content = [" ".join(text) for text in sample["body_text"]]
                    summary = [summ for summ in sample["abstract"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_elsevier_data_after_downloading,
                                                                     batched=True)

            elif self.dataset_name == "scitldr":
                def process_scitldr_data_after_downloading(sample):
                    content = [" ".join(text) for text in sample["source"]]
                    summary = [" ".join(summ) for summ in sample["target"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_scitldr_data_after_downloading,
                                                                     batched=True)

            elif self.dataset_name == "cord19":
                def process_cord19_data_after_downloading(sample):
                    content = [text for text in sample["fulltext"]]
                    summary = [summ for summ in sample["abstract"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                train_test_val_split = loaded_dataset.train_test_split(test_size=0.05)
                train_val_split = train_test_val_split["train"].train_test_split(test_size=0.1667)
                loaded_dataset = DatasetDict({"train": train_val_split["train"],
                                              "validation": train_val_split["test"],
                                              "test": train_test_val_split["test"]})
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_cord19_data_after_downloading, batched=True)

            elif self.dataset_name == "sci_lay":
                def process_sci_lay_data_after_downloading(sample):
                    content = [text for text in sample["full_text"]]
                    summary = [summ for summ in sample["plain_text"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                process_data = {}
                for split in ["train", "validation", "test"]:
                    process_data[split] = Dataset.from_list(list(loaded_dataset[split]))
                loaded_dataset = DatasetDict(process_data)
                del process_data
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_sci_lay_data_after_downloading, batched=True)

            elif self.dataset_name == "mslr":
                def process_mslr_data_after_downloading(sample):
                    content = [" ".join(text) for text in sample["full_text"]]
                    summary = [summ for summ in sample["plain_text"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                process_data = {}
                for split in ["train", "validation", "test"]:
                    process_data[split] = Dataset.from_list(list(loaded_dataset[split]))
                loaded_dataset = DatasetDict(process_data)
                del process_data
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_mslr_data_after_downloading, batched=True)

            elif self.dataset_name == "eur_lex":
                def process_eur_lex_data_after_downloading(sample):
                    content = [text for text in sample["reference"]]
                    summary = [summ for summ in sample["summary"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                process_data = {}
                for split in ["train", "validation", "test"]:
                    process_data[split] = Dataset.from_list(list(loaded_dataset[split]))
                loaded_dataset = DatasetDict(process_data)
                del process_data
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_eur_lex_data_after_downloading, batched=True)

            elif self.dataset_name == "bill_sum":
                def process_bill_sum_data_after_downloading(sample):
                    content = [text for text in sample["text"]]
                    summary = [summ for summ in sample["summary"]]
                    return {
                        "content": content,
                        "summary": summary
                    }

                loaded_dataset = load_dataset(path=self.dataset_info.dataset_id, name=self.dataset_info.version,
                                              streaming=self.dataset_info.streaming,
                                              trust_remote_code=self.dataset_info.trust_remote_code)
                process_data, test_data = {}, []
                for split in ["train", "ca_test", "test"]:
                    if split == "train":
                        process_data[split] = Dataset.from_list(list(loaded_dataset[split]))
                    else:
                        if split == "ca_test":
                            test_data.extend(list(loaded_dataset[split]))
                        else:
                            test_data.extend(list(loaded_dataset[split]))
                            process_data["test"] = Dataset.from_list(test_data)

                train_val_split = process_data["train"].train_test_split(test_size=0.2)
                loaded_dataset = DatasetDict({"train": train_val_split["train"],
                                              "validation": train_val_split["test"],
                                              "test": process_data["test"]})
                del process_data, train_val_split
                loaded_dataset = loaded_dataset.shuffle(seed=42).map(process_bill_sum_data_after_downloading, batched=True)

            loaded_dataset = loaded_dataset.remove_columns(self.dataset_info.columns_to_remove)
            self.train_set = Dataset.from_list(self.sample_dataset(loaded_dataset["train"],
                                                                   sample_size=min(self.training_samples,
                                                                                   loaded_dataset["train"].num_rows)))
            self.validation_set = Dataset.from_list(self.sample_dataset(loaded_dataset["validation"],
                                                                        sample_size=min(self.eval_samples,
                                                                                        loaded_dataset["validation"].num_rows)))
            self.test_set = Dataset.from_list(self.sample_dataset(loaded_dataset["test"],
                                                                  sample_size=min(self.test_samples,
                                                                                  loaded_dataset["test"].num_rows)))

            loaded_dataset = DatasetDict({
                "train": self.train_set,
                "validation": self.validation_set,
                "test": self.test_set
            })

            loaded_dataset.save_to_disk(local_path)
            print("******* '{}' Dataset for {} Domain -  is stored at {} !!!*******".format(self.dataset_name,
                                                                                            self.domain.name,
                                                                                            local_path))

        else:
            loaded_dataset = load_from_disk(local_path)

        loaded_dataset = loaded_dataset.map(lambda x: {"content_tokens": len(x["content"].split()),
                                                       "summary_tokens": len(x["summary"].split())})

        loaded_dataset = loaded_dataset.filter(lambda x: (len(x["content"]) > 0 and len(x["summary"]) > 0))

        print("Min and Max number of tokens in articles in train dataset of {} are {} and {} "
              "respectively!".format(self.dataset_name, min(loaded_dataset["train"]["content_tokens"]),
                                     max(loaded_dataset["train"]["content_tokens"])))

        if self.sort_dataset_on_article_len:
            print("Sorting test data on number of tokens of articles")
            loaded_dataset = loaded_dataset.sort("content_tokens")
        loaded_dataset = loaded_dataset.remove_columns(["content_tokens", "summary_tokens"])
        self.train_set = loaded_dataset["train"].select(range(min(self.training_samples, len(loaded_dataset["train"]))))
        self.validation_set = loaded_dataset["validation"].select(
            range(min(self.eval_samples, len(loaded_dataset["validation"]))))
        self.test_set = loaded_dataset["test"].select(range(min(self.test_samples, len(loaded_dataset["test"]))))
        del loaded_dataset

        self.return_stats()

        # return self.train_set, self.validation_set, self.test_set

    def loading_dataset_from_csv_or_excel(self):
        # TODO: To be implemented
        pass
        raise NotImplementedError()
        # return None, None, None

    def loading_dataset_splits(self):
        if self.dataset_info.source in ["hugging_face", ".tar", ".zip", ".tar.gz"]:
            return self.loading_dataset_from_hf()

        elif self.dataset_info.source in [".csv", ".xlsx"]:
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
        cols_to_remove = [col for col in self.train_set.column_names if
                          col not in ["input_ids", "attention_mask", "labels"]]
        self.train_set = self.train_set.map(tokenization_process, batched=True, remove_columns=cols_to_remove)
        self.validation_set = self.validation_set.map(tokenization_process, batched=True, remove_columns=cols_to_remove)
        return self.train_set, self.validation_set, self.test_set
