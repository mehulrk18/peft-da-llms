from dataset_lib import SumDataLoader


def download_dataset_and_store(domain_name: str, dataset_name: str, training_samples: int = 5000,
                               validation_samples: int = 1000, testing_samples: int = 1000,
                               preprocess_function: callable = None, sort_dataset_on_article_len: bool = False):
    # try:
    data = SumDataLoader(domain=domain_name, dataset_name=dataset_name, training_samples=training_samples,
                         eval_samples=validation_samples, test_samples=testing_samples,
                         sort_dataset_on_article_len=sort_dataset_on_article_len, force_download=True)

    data.loading_dataset_splits()

    data.train_set = data.processing_data_with_training_prompt(data.train_set,
                                                               preprocess_function=preprocess_function)
    data.validation_set = data.processing_data_with_training_prompt(data.validation_set,
                                                                    preprocess_function=preprocess_function)

    print("**Dataset after it is processed with Prompt**\n\n{}".format(data))
    print("!! Dataset {} from domain {} is processed with Prompt and now saving it to disk at : {}".format(
        data.dataset_name, data.domain, data.dataset_info.local_path))

    # except Exception as e:
    #     print("!!!!!! CANNOT GENERATE DATASET BECAUSE BELOW EXCEPTION CAUGHT WHILE PROCESSING DATASET - '{}' "
    #           "FROM DOMAIN '{}' !!!!!!\n\n\nEXCEPTION: {} \n".format(dataset_name, domain_name, e))
