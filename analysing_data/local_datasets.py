import logging
from typing import Mapping, Sequence, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from dataset_lib import datasets_info_dict


class LoadDatasetFromLocal:
    ds: DatasetDict
    domain_name: str
    ds_name: str

    def __init__(
        self,
        # ds_subset: str,
        preview: bool,
        remove_columns: list = None,
        col_map: dict = None,
        preview_size: int = 4,
        min_input_size: int = 0,
        samples: Union[int, str] = "max",
        # load_csv: bool = False,
        data_files: Union[
            dict, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None
        ] = None,
    ) -> None:

        print("DOMAIN_NAME:", self.domain_name)
        print("DATASET_NAME:", self.ds_name)
        from dataset_lib import SumDomains
        self.dataset_info = datasets_info_dict[SumDomains(self.domain_name)][self.ds_name]

        data = load_from_disk(self.dataset_info["local_path"])

        if preview:
            for k in data.keys():
                data[k] = data[k].select(range(preview_size))

        elif samples != "max":
            data["train"] = data["train"].select(
                range(min(len(data["train"]), samples))
            )

        self.ds = self.preprocess(
            data, col_map, min_input_size, remove_columns=[] if remove_columns is None else remove_columns
        )

    def get_split(self, key: str) -> Dataset:
        return self.ds[key]

    @classmethod
    def scrolls_preprocess(
        cls,
        data: DatasetDict,
        col_map: dict,
        min_input_size: int,
        remove_columns: list,
    ) -> DatasetDict:
        # conditions for admitting data into the training:
        # 1) Text (x) is twice as long as summary (y) and less than 1000 times longer.
        # 2) Summary is not a verbatim part of the text.
        # 3) The text document has a minimum length (min_input_size).
        def mask(x, y):
            return (
                2 * len(y) < len(x) < 1000 * len(y)
                and y not in x
                and len(x) >= min_input_size
            )

        def none_data_filter(example):
            return example["text"] is not None and example["summary"] is not None

        def fn(batch: dict):
            res = {"text": [], "summary": []}
            z = zip(batch["text"], batch["summary"])
            # apply the logical inverse of `mask` to obtain admissible documents.
            valid = list(filter(lambda x: mask(x[0], x[1]), z))
            # print(valid)
            res["text"] = [valid[idx][0] for idx in range(len(valid))]
            res["summary"] = [valid[idx][1] for idx in range(len(valid))]
            return res

        logging.info("Preprocessing dataset")
        data = data.rename_columns(col_map)

        # save_test = data["test"]
        #print(data)

        data = data.filter(none_data_filter)
        # print(len(data["text"]), len(data["summary"]))
        if remove_columns == []:
            data = data.map(
                fn,
                batched=True,
            )
        else:
            data = data.map(fn, batched=True, remove_columns=remove_columns)

        # data["test"] = save_test
        data.set_format("torch")
        return data

    def combine_document_field(
        self, dataset: Dataset, document_field: str = "sources"
    ) -> Dataset:
        def combine_strings(example):
            if isinstance(example[document_field], list):
                example[document_field] = " ".join(example[document_field])
            return example

        # Apply the function to the dataset
        updated_dataset = dataset.map(combine_strings)

        return updated_dataset

    def combine_columns(
        self, dataset: Dataset, document_field: str = "sources"
    ) -> Dataset:
        def combine_strings(example):
            example["summary"] = (
                f"{example['challenge']} \n {example['approach']} \n {example['outcome']}"
            )
            return example

        # Apply the function to the dataset
        updated_dataset = dataset.map(combine_strings)

        return updated_dataset

    def preprocess(
        self,
        data: DatasetDict,
        col_map: dict,
        min_input_size: int,
        remove_columns: list,
    ) -> DatasetDict:
        # subclasses can implement custom behaviour by defining the preprocess fn
        return self.scrolls_preprocess(
            data=data,
            col_map=col_map,
            min_input_size=min_input_size,
            remove_columns=remove_columns,
        )


class Arxiv(LoadDatasetFromLocal):
    ds_name = "arxiv"
    domain_name = "scientific"
    dataset_kwargs = {
        # "ds_name": "ccdv/arxiv-summarization",
        # "ds_subset": "document",
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            data_files=data_files,
            **self.dataset_kwargs,
        )


class Pubmed(LoadDatasetFromLocal):
    ds_name = "pubmed"
    domain_name = "medical"
    dataset_kwargs = {
        # "ds_name": "ccdv/pubmed-summarization",
        # "ds_subset": "document",
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            **self.dataset_kwargs,
        )


class MultiLex(LoadDatasetFromLocal):
    ds_name = "multilex"
    domain_name = "legal"
    dataset_kwargs = {
        # "ds_name": "allenai/multilexsum",
        # "ds_subset": "v20230518",
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [
            "sources_metadata",
            "summary/short",
            "summary/tiny",
            "id",
            "case_metadata",
        ],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        # load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            data_files=data_files,
            **self.dataset_kwargs,
        )


class CNNDailyMail(LoadDatasetFromLocal):
    ds_name = "cnndm"
    domain_name = "news"
    dataset_kwargs = {
        # "ds_name": "abisee/cnn_dailymail",
        # "ds_subset": "1.0.0",  # other options: "2.0.0" and  "3.0.0"
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            **self.dataset_kwargs,
        )


class MultiNews(LoadDatasetFromLocal):
    ds_name = "multinews"
    domain_name = "news"
    dataset_kwargs = {
        # "ds_name": "abisee/cnn_dailymail",
        # "ds_subset": "1.0.0",  # other options: "2.0.0" and  "3.0.0"
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            **self.dataset_kwargs,
        )


class XSumNews(LoadDatasetFromLocal):
    ds_name = "xsum"
    domain_name = "news"
    dataset_kwargs = {
        # "ds_name": "abisee/cnn_dailymail",
        # "ds_subset": "1.0.0",  # other options: "2.0.0" and  "3.0.0"
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            **self.dataset_kwargs,
        )

class NewsRoom(LoadDatasetFromLocal):
    ds_name = "newsroom"
    domain_name = "news"
    dataset_kwargs = {
        # "ds_name": "lil-lab/newsroom",
        # "ds_subset": "default",
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [
            # "title",
            # "url",
            # "date",
            # "density",
            # "coverage",
            # "compression",
            # "density_bin",
            # "coverage_bin",
            # "compression_bin",
        ],
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
        # load_csv: bool = False,
        data_files=None,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            # load_csv=load_csv,
            **self.dataset_kwargs,
        )