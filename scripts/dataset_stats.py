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
        for _dataset in datasets:
            data = SumDataLoader(
                domain=domain,
                dataset_name=_dataset,
                training_samples=samples["train"],
                eval_samples=samples["validation"],
                test_samples=samples["test"],
                sort_dataset_on_article_len=True
            )

            # TODO: write a lambda function for counting avg tokens
            data.train_set = data.train_set.map(count_tokens, batched=True)
            data.validation_set = data.validation_set.map(count_tokens, batched=True)
            data.test_set = data.test_set.map(count_tokens, batched=True)

            print(f"**** Domain: {domain} ---  Dataset: {_dataset} ***")
            print("\n\nAvg Tokens Articles in Training Set:", data.train_set["content_tokens"].mean())
            print("Avg Tokens Summary in Training Set:", data.train_set["summary_tokens"].mean())

            print("\n\nAvg Tokens Articles in Validation Set:", data.validation_set["content_tokens"].mean())
            print("Avg Tokens Summary in Validation Set:", data.validation_set["summary_tokens"].mean())

            print("\n\nAvg Tokens Articles in Test Set:", data.test_set["content_tokens"].mean())
            print("Avg Tokens Summary in Test Set:", data.test_set["summary_tokens"].mean())


            