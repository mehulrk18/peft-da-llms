from dataset_lib import preprocessing_data_with_prompt
from utils.fetch_data_and_store import download_dataset_and_store

if __name__ == "__main__":
    download_dataset_and_store(domain_name="medical", dataset_name="pubmed", training_samples=10000,
                               validation_samples=2000, testing_samples=1000, sort_dataset_on_article_len=False,
                               preprocess_function=preprocessing_data_with_prompt)
