import pandas as pd

from dataset_lib import datasets_info_dict, SumDataLoader
from utils import LLaMAModelClass


def calculate_stats(llama, samples, dest_file: str):
    data_statss = []
    for domain, datasets in datasets_info_dict.items():
        for _dataset in datasets.keys():
            train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            if domain.name.lower() == "unseen_test":
                p = datasets_info_dict[domain][_dataset]
                print(p.local_path)
                if p.local_path.endswith(".xlsx"):
                    data = pd.read_excel(p.local_path)
                else:
                    data = pd.read_csv(p.local_path)
                test_df["content_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data["content"].tolist()]
                test_df["summary_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data["summary"].tolist()]
            else:
                data = SumDataLoader(
                    domain=domain.name.lower(),
                    dataset_name=_dataset,
                    training_samples=samples["train"],
                    eval_samples=samples["validation"],
                    test_samples=samples["test"],
                    sort_dataset_on_article_len=True
                )

                train_df["content_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data.train_set["content"]]
                train_df["summary_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data.train_set["summary"]]

                val_df["content_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data.validation_set["content"]]
                val_df["summary_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data.validation_set["summary"]]

                test_df["content_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data.test_set["content"]]
                test_df["summary_tokens"] = [len(llama.tokenizer.tokenize(x)) for x in data.test_set["summary"]]

            print(f"**** Domain: {domain} ---  Dataset: {_dataset} ***")
            obj = {
                "domain": domain,
                "dataset": _dataset,
                "train_samples": len(train_df),
                "train_average_content_tokens": train_df["content_tokens"].mean() if len(train_df) > 0 else 0,
                "train_average_summary_tokens": train_df["summary_tokens"].mean() if len(train_df) > 0 else 0,
                "validation_samples": len(val_df),
                "validation_average_content_tokens": val_df["content_tokens"].mean() if len(val_df) > 0 else 0,
                "validation_average_summary_tokens": val_df["summary_tokens"].mean() if len(val_df) > 0 else 0,
                "test_samples": len(test_df),
                "test_average_content_tokens": test_df["content_tokens"].mean() if len(test_df) > 0 else 0,
                "test_average_summary_tokens": test_df["summary_tokens"].mean() if len(test_df) > 0 else 0
            }
            data_statss.append(obj)
            # print("\n\nAvg Tokens Articles in Training Set:", train_df["content_tokens"].mean())
            # print("Avg Tokens Summary in Training Set:", train_df["summary_tokens"].mean())
            #
            # print("\n\nAvg Tokens Articles in Validation Set:", val_df["content_tokens"].mean())
            # print("Avg Tokens Summary in Validation Set:", val_df["summary_tokens"].mean())
            #
            # print("\n\nAvg Tokens Articles in Test Set:", test_df["content_tokens"].mean())
            # print("Avg Tokens Summary in Test Set:", test_df["summary_tokens"].mean())
    print("Saving statistics to file: ", dest_file)
    df = pd.DataFrame(data_statss)
    df.to_excel(dest_file, index=False)


if __name__ == "__main__":
    used_samples = {
        "train": 1000,
        "validation": 500,
        "test": 150
    }
    max_samples = {
        "train": 10000,
        "validation": 5000,
        "test": 5000
    }
    llama = LLaMAModelClass(
        instruct_mode=True,
        mlm=True
    )

    calculate_stats(llama, used_samples, "results/dataset_used_samples_stats.xlsx")
    calculate_stats(llama, max_samples, "results/dataset_max_samples_stats.xlsx")
