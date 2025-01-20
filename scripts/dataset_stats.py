import pandas as pd

from dataset_lib import datasets_info_dict, SumDataLoader
from utils import LLaMAModelClass

if __name__ == "__main__":
    samples = {
        "train": 1000,
        "validation": 500,
        "test": 150
    }
    llama = LLaMAModelClass(
        instruct_mode=True,
        mlm=True
    )

    def count_tokens(sample):
        content_tokens = len(llama.tokenizer(sample["content"], return_tensors="pt"))
        summary_tokens = len(llama.tokenizer(sample["summary"], return_tensors="pt"))

        return {"content_tokens": content_tokens, "summary_tokens": summary_tokens}

    for domain, datasets in datasets_info_dict.items():
        for _dataset in datasets.keys():
            if domain.name.lower() == "unseen_test":
                if datasets_info_dict[domain][_dataset].local_path.endswith(".xlsx"):
                    data = pd.read_excel(_dataset.local_path)
                else:
                    data = pd.read_csv(_dataset.local_path)
                test_df = pd.DataFrame()
                test_df["content_tokens"] = data["content"].apply(lambda x: len(llama.tokenizer.tokenize(x)))
                test_df["summary_tokens"] = data["summary"].apply(lambda x: len(llama.tokenizer.tokenize(x)))
            else:
                data = SumDataLoader(
                    domain=domain.name.lower(),
                    dataset_name=_dataset,
                    training_samples=samples["train"],
                    eval_samples=samples["validation"],
                    test_samples=samples["test"],
                    sort_dataset_on_article_len=True
                )
                train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

                # TODO: write a lambda function for counting avg tokens
                # data.train_set = data.train_set.map(count_tokens, batched=True)
                # data.validation_set = data.validation_set.map(count_tokens, batched=True)
                # data.test_set = data.test_set.map(count_tokens, batched=True)

                train_df["content_tokens"] = data.train_set["content"].apply(lambda x: len(llama.tokenizer.tokenize(x)))
                train_df["summary_tokens"] = data.train_set["summary"].apply(lambda x: len(llama.tokenizer.tokenize(x)))

                val_df["content_tokens"] = data.validation_set["content"].apply(lambda x: len(llama.tokenizer.tokenize(x)))
                val_df["summary_tokens"] = data.validation_set["summary"].apply(lambda x: len(llama.tokenizer.tokenize(x)))

                test_df["content_tokens"] = data.test_set["content"].apply(lambda x: len(llama.tokenizer.tokenize(x)))
                test_df["summary_tokens"] = data.test_set["summary"].apply(lambda x: len(llama.tokenizer.tokenize(x)))

                print(f"**** Domain: {domain} ---  Dataset: {_dataset} ***")
                print("\n\nAvg Tokens Articles in Training Set:", train_df["content_tokens"].mean())
                print("Avg Tokens Summary in Training Set:", train_df["summary_tokens"].mean())

                print("\n\nAvg Tokens Articles in Validation Set:", val_df["content_tokens"].mean())
                print("Avg Tokens Summary in Validation Set:", val_df["summary_tokens"].mean())

            print("\n\nAvg Tokens Articles in Test Set:", test_df["content_tokens"].mean())
            print("Avg Tokens Summary in Test Set:", test_df["summary_tokens"].mean())


            