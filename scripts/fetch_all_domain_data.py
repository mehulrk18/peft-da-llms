import os

from dataset_lib import preprocessing_data_with_prompt, datasets_info_dict
from utils.fetch_data_and_store import download_dataset_and_store

if __name__ == "__main__":

    for domain, dataset_obj in datasets_info_dict.items():
        # for dataset_name, values in datasets.items():
        try:
            if os.path.exists(dataset_obj.local_path):
                print("!! Dataset - '{}' from Domain - '{}' already exists at path {} hence, skipping it. "
                      "!!".format(dataset_obj.name, domain.name, dataset_obj.local_path))
                continue
            download_dataset_and_store(domain_name=str(domain.name), dataset_name=dataset_obj.name,
                                       training_samples=10000, validation_samples=2000, testing_samples=1000,
                                       sort_dataset_on_article_len=False,
                                       preprocess_function=preprocessing_data_with_prompt)
        except Exception as e:
            print("!! Dataset - '{}' from Domain - '{}' has not information available hence, "
                  "skipping it. !!".format(dataset_obj.name, domain.name))


