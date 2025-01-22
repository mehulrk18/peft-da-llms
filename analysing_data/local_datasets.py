import logging
import pandas as pd
import dataset_lib
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

        if self.dataset_info.local_path.endswith(".csv"):
            data = pd.read_csv(self.dataset_info.local_path)
            data = data.drop(columns=[col for col in data.columns if not col in ["content", "summary"]])
            # data.rename(columns={"content": "text"}, inplace=True)
            data = DatasetDict({"test": Dataset.from_pandas(data)})

        elif self.dataset_info.local_path.endswith(".xlsx"):
            data = pd.read_excel(self.dataset_info.local_path)
            data = data.drop(columns=[col for col in data.columns if not col in ["content", "summary"]])
            # data.rename(columns={"content": "text"}, inplace=True)
            data = DatasetDict({"test": Dataset.from_pandas(data)})
        else:
            data = load_from_disk(self.dataset_info.local_path)
            data = data.filter(lambda x: len(x['content']) > 0 and len(x['summary']) > 0)
            loaded_dataset = data.map(lambda x: {"content_len": len(x["content"])})
            data = loaded_dataset.sort("content_len")
            data = data.remove_columns(["content_len"])
            

        if preview:
            samples = {
                "train": 1000,
                "validation": 1,
                "test": 150
            }
            for k in data.keys():
                # data[k] = data[k].select(range(preview_size))
                data[k] = data[k].select(range(samples[k]))

        elif samples != "max":
            if "train" in list(data.keys()):
                data["train"] = data["train"].select(
                    range(min(len(data["train"]), samples))
                )

        self.ds = self.preprocess(
            data, col_map, min_input_size, remove_columns=[] if remove_columns is None else remove_columns
        )

    def get_split(self, key: str) -> Dataset:
        if key in list(self.ds.keys()):
            return self.ds[key]
        else:
            raise ValueError(f"Key {key} not found in dataset.")

    def scrolls_preprocess(
        self,
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


class LocalDatasetArxiv(LoadDatasetFromLocal):
    ds_name = "arxiv"
    domain_name = dataset_lib.SumDomains.SCIENTIFIC.name.lower() #"scientific"
    dataset_kwargs = {
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


class LocalDatasetElsevier(LoadDatasetFromLocal):
    ds_name = "elsevier"
    domain_name = dataset_lib.SumDomains.SCIENTIFIC.name.lower() #"scientific"
    dataset_kwargs = {
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


class LocalDatasetSciTLDR(LoadDatasetFromLocal):
    ds_name = "scitldr"
    domain_name = dataset_lib.SumDomains.SCIENTIFIC.name.lower() #"scientific"
    dataset_kwargs = {
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


class LocalDatasetPubmed(LoadDatasetFromLocal):
    ds_name = "pubmed"
    domain_name = dataset_lib.SumDomains.MEDICAL.name.lower() # "medical"
    dataset_kwargs = {
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


class LocalDatasetCord19(LoadDatasetFromLocal):
    ds_name = "cord19"
    domain_name = dataset_lib.SumDomains.MEDICAL.name.lower() # "medical"
    dataset_kwargs = {
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

class LocalDatasetMSLR(LoadDatasetFromLocal):
    ds_name = "mslr"
    domain_name = dataset_lib.SumDomains.MEDICAL.name.lower() # "medical"
    dataset_kwargs = {
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


class LocalDatasetSciLay(LoadDatasetFromLocal):
    ds_name = "scilay"
    domain_name = dataset_lib.SumDomains.MEDICAL.name.lower() # "medical"
    dataset_kwargs = {
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


class LocalDatasetMultiLex(LoadDatasetFromLocal):
    ds_name = "multilex"
    domain_name = dataset_lib.SumDomains.LEGAL.name.lower() #"legal"
    dataset_kwargs = {
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
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


class LocalDatasetEurLex(LoadDatasetFromLocal):
    ds_name = "eurlex"
    domain_name = dataset_lib.SumDomains.LEGAL.name.lower() #"legal"
    dataset_kwargs = {
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
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


class LocalDatasetBillSum(LoadDatasetFromLocal):
    ds_name = "billsum"
    domain_name = dataset_lib.SumDomains.LEGAL.name.lower() #"legal"
    dataset_kwargs = {
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
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


class LocalDatasetCNNDailyMail(LoadDatasetFromLocal):
    ds_name = "cnndm"
    domain_name = dataset_lib.SumDomains.NEWS.name.lower()
    dataset_kwargs = {
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


class LocalDatasetMultiNews(LoadDatasetFromLocal):
    ds_name = "multinews"
    domain_name = dataset_lib.SumDomains.NEWS.name.lower()
    dataset_kwargs = {
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


class LocalDatasetXSumNews(LoadDatasetFromLocal):
    ds_name = "xsum"
    domain_name = dataset_lib.SumDomains.NEWS.name.lower()
    dataset_kwargs = {
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


class LocalDatasetNewsRoom(LoadDatasetFromLocal):
    ds_name = "newsroom"
    domain_name = dataset_lib.SumDomains.NEWS.name.lower()
    dataset_kwargs = {
        # "ds_name": "lil-lab/newsroom",
        # "ds_subset": "default",
        "col_map": {"content": "text", "summary": "summary"},
        "remove_columns": [],
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


class LocalDatasetScientific(LoadDatasetFromLocal):
    ds_name = dataset_lib.SumDomains.SCIENTIFIC.name.lower()
    domain_name = dataset_lib.SumDomains.UNSEEN_TEST.name.lower()
    dataset_kwargs = {
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


class LocalDatasetMedical(LoadDatasetFromLocal):
    ds_name = dataset_lib.SumDomains.MEDICAL.name.lower()
    domain_name = dataset_lib.SumDomains.UNSEEN_TEST.name.lower()
    dataset_kwargs = {
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


class LocalDatasetLegal(LoadDatasetFromLocal):
    ds_name = dataset_lib.SumDomains.LEGAL.name.lower()
    domain_name = dataset_lib.SumDomains.UNSEEN_TEST.name.lower()
    dataset_kwargs = {
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


class LocalDatasetNews(LoadDatasetFromLocal):
    ds_name = dataset_lib.SumDomains.NEWS.name.lower()
    domain_name = dataset_lib.SumDomains.UNSEEN_TEST.name.lower()
    dataset_kwargs = {
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