from dataset_lib import preprocessing_data_with_prompt, datasets_info_dict
from utils.fetch_data_and_store import download_dataset_and_store

if __name__ == "__main__":

    for domain, datasets in datasets_info_dict.items():
        for dataset_name, values in datasets.items():
            if not values:
                print("!! Dataset - '{}' from Domain - '{}' has not information available hence, "
                      "skipping it. !!".format(dataset_name, domain.name))
                continue
            download_dataset_and_store(domain_name=str(domain.name), dataset_name=dataset_name,
                                       training_samples=10000, validation_samples=2000, testing_samples=1000,
                                       sort_dataset_on_article_len=False,
                                       preprocess_function=preprocessing_data_with_prompt)


