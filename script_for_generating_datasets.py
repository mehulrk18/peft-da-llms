from dataset_lib import SumDataLoader, SumDatasets


def generate_datasets(name, force_download=False, training_samples=5000):
    try:
        data = SumDataLoader(dataset_name=name, training_samples=training_samples, force_download=force_download)

        data.loading_dataset_from_hf()

        data.train_set = data.processing_data_with_prompt(data.train_set)
        data.validation_set = data.processing_data_with_prompt(data.validation_set)

        print("**Dataset after it is processed with Prompt**\nTrain: {}\n\nValidation: {}\n\n".format(data.train_set, data.validation_set))
        del data

    except Exception as e:
        print("!!!!!! CANNOT GENERATE DATASET FOR {} BECAUSE BELOW EXCEPTION CAUGHT !!!!!!\n\n\n{}".format(name, e))


if __name__ == "__main__":
    data_dict = dict(SumDatasets.__members__)
    print("Script to forcefully generate random samples of all dataset's available.")
    for value in data_dict.values():
        generate_datasets(name=value.value, force_download=True)
