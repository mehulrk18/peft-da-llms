import os
import random

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

from .domains_utils import datasets_info_dict, SumDatasets, preprocessing_scientific_or_medical, \
    preprocessing_legal, preprocessing_news, preprocessing_low_resource_domain

DATASET_STORAGE_DIR = "domains/"   # fetch from configs


class SumDataLoader:

    def __init__(self, dataset_name: str, training_samples: int = 1000, force_download: bool = False):

        self.dataset = SumDatasets(dataset_name.lower())
        self.dataset_name = self.dataset.name

        self.force_download = force_download
        self.training_samples = training_samples

        self.dataset_id = datasets_info_dict[self.dataset_name]["dataset_id"]
        self.local_path = datasets_info_dict[self.dataset_name]["local_path"]
        self.dataset_source = datasets_info_dict[self.dataset_name]["source"]

        self.train_set, self.validation_set, self.test_set = None, None, None

        self.preprocess_function = self.get_preprocess_function()

    @classmethod
    def sample_dataset(cls, dataset_split, sample_size=1000):
        random.seed(42)
        # dataset_split = dataset_split.to_dataset()
        print("datasplit: ", dataset_split)
        # import pdb; pdb.set_trace()

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
        local_path = DATASET_STORAGE_DIR + self.local_path
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
            # del dataset_dict
            # return _dataset
        else:
            loaded_dataset = load_from_disk(local_path)
            self.train_set = loaded_dataset["train"]
            self.validation_set = loaded_dataset["validation"]
            self.test_set = loaded_dataset["test"]

        return self.train_set, self.validation_set, self.test_set

    def loading_dataset_from_csv_or_excel(self):
        # TODO: To be implemented
        pass
        return None, None, None

    def loading_dataset_splits(self):
        if self.dataset_source == "hugging_face":
            return self.loading_dataset_from_hf()

        elif self.dataset_source in ["csv_file", "excel_file"]:
            return self.loading_dataset_from_csv_or_excel()

        else:
            NotImplementedError("Sources except hugging_face and tabular (csv or excel) NOT IMPLEMENTED")

    def get_preprocess_function(self):
        if self.dataset_name in [SumDatasets.ARXIV.name, SumDatasets.PUBMED.name]:
            return preprocessing_scientific_or_medical

        elif self.dataset_name == SumDatasets.MULTI_LEX:
            return preprocessing_legal

        elif self.dataset_name == SumDatasets.NEWS:
            return preprocessing_news

        else:
            return preprocessing_low_resource_domain

    def processing_data_with_prompt(self, dataset_split: Dataset):
        return (dataset_split.map(self.preprocess_function, batched=True
                                  ).shuffle(seed=42))

    def tokenization_of_data_splits(self, tokenization_process: callable):
        self.train_set = self.train_set.map(tokenization_process, batched=True)
        self.train_set = self.train_set.remove_columns(
            [col for col in self.train_set.column_names if col != "input_ids"])

        self.validation_set = self.validation_set.map(tokenization_process, batched=True)
        self.validation_set = self.validation_set.remove_columns(
            [col for col in self.validation_set.column_names if col != "input_ids"])

        return self.train_set, self.validation_set, self.test_set
