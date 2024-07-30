import pandas as pd
import huggingface_hub
from datasets import list_datasets, load_dataset


def read_scitldr(directory):
    dir_path = "domains/{}/".format(directory)

    splits = ["train", "val", "test"]

    for sp in splits:
        # with open("{}{}.csv".format(dir_path, sp), "rb") as f:

        df = pd.read_csv("{}{}.csv".format(dir_path, sp))

        import pdb; pdb.set_trace()

        print(sp, "\n", df[:6])


def read_arxiv_data():
    # for x in huggingface_hub.list_datasets():
    #     if "arxiv" in x.id:
    #         print(x)
    #         print("\n")

    dataset = load_dataset(path='ccdv/arxiv-summarization', streaming=True, trust_remote_code=True)

    # import pdb; pdb.set_trace()
    return dataset


def read_multi_lexsum():
    dataset = load_dataset(path='allenai/multi_lexsum', name='v20230518', streaming=True, trust_remote_code=True)

    print("Dataset: ", dataset)

    splits = {
        "train": list(dataset["train"]),
        "validation": list(dataset["validation"]),
        "test": list(dataset["test"]),
    }

    return dataset  # splits
    # exp = list(dataset["validation"])[4]
    #
    # print(exp["sources"])
    #
    # for sum_len in ["long", "short", "tiny"]:
    #     print(exp["summary/" + sum_len])  # Summaries of three lengths
    #
    # print(exp['case_metadata'])  # The cor

    # import pdb; pdb.set_trace()


def read_news_summarization():
    file_path = "domains/news_summarization.csv"

    df = pd.read_csv(file_path)

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    # read_scitldr("SciTLDR-A")
    read_news_summarization()

