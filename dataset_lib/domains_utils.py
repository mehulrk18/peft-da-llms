from enum import Enum

import pandas as pd
import huggingface_hub
from datasets import list_datasets, load_dataset

from .prompt_utils import generate_training_prompt


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


def preprocessing_scientific_or_medical(sample):
    texts = [generate_training_prompt(article=article, summary=summary)
             for article, summary in zip(sample["article"], sample["abstract"])]

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


if __name__ == "__main__":
    # read_scitldr("SciTLDR-A")
    read_news_summarization()

