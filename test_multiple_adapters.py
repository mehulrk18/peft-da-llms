import argparse
import logging

import pandas as pd
import torch

from dataset_lib import SumDataLoader
from testing_one_v_one_model import testing_model
from utils import generate_summary, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, read_yaml, torch_dtypes_dict


global CHAT_TEMPLATE


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    global CHAT_TEMPLATE

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--trained_adapter_config_file", type=str, required=True,
                        help="yaml file containing configs and paths of trained adapters.")

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        main_directory = ""

    args = parser.parse_args()
    trained_peft_config_path = main_directory + args.trained_adapter_config_file
    testing_configs = read_yaml(file_name=trained_peft_config_path)
    mlm = testing_configs["mlm"]
    training_samples = testing_configs["training_samples"]
    eval_samples = testing_configs["eval_samples"]
    test_samples = testing_configs["test_samples"]
    sort_data = testing_configs["sorted_dataset"]
    CHAT_TEMPLATE = testing_configs["chat_template"]
    use_instruct_model = testing_configs["instruct_mode"]
    provider = testing_configs["provider"]
    metric = testing_configs["metric"]
    test_domain = testing_configs["test_domain"]
    test_dataset_name = testing_configs["test_dataset_name"]
    torch_dtype = torch_dtypes_dict[testing_configs["torch_dtype"]]
    quantize = testing_configs.get("torch_dtype", False)

    adapter_paths = testing_configs["trained_adapters_path"]

    adapter_names = []

    for path in adapter_paths:
        a_name = path.split("/")[-1]
        adapter_names.append(a_name)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    all_adapters = "-".join(adapter_names)
    logging.basicConfig(
        filename=main_directory + 'logs/testing_{}_{}samples.log'.format(all_adapters, test_samples),  # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    logger = logging.getLogger()
    logger.info("Device in use: {}".format(device))

    # llama_model = get_pretrained_model(ah=ah)
    llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize, mlm=mlm,
                            torch_dtype=torch_dtype)
    # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

    logger.info("Check point MODEL: \n{}".format(llama.model))

    if provider == "hf":
        # Method 1 - HuggingFace
        from peft import PeftModel
        for a_path, a_name in zip(adapter_paths, adapter_names):
            llama.model.load_adapter(a_path, adapter_name="trained_"+a_name)

        for i in range(len(adapter_names)):
            adapter_names[i] = "trained_" + adapter_names[i]
        # llama.model.load_adapter(lora_path, adapter_name="m_lora")
        # llama.model = PeftModel.from_pretrained(llama.model, trained_peft_path, adapter_name=adapter_name) #, use_safetensors=True)
        # # llama.model = llama.model.merge_and_unload()
        # llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
        llama.model.set_adapter([adapter_names])
        llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name="trained_", torch_dtype=torch_dtype)
        llama.model = llama.model.to(torch_dtype)
    # else:
    #     # Method 2 - AdapterHub
    #     adapters.init(model=llama.model)
    #     loaded_peft = llama.model.load_adapter(trained_peft_path, with_head=False)
    #     llama.model.set_active_adapters([loaded_peft])
    #     llama.model.adapter_to(loaded_peft, device=device)
    #
    #     llama.model = llama.model.to(torch.bfloat16)
    #
    #     llama.model.enable_input_require_grads()
    #     llama.model.gradient_checkpointing_enable()
    #     logger.info("\nMethod 2 LLaMA Model's Summary:\n{}\n\n\n".format(llama.model.adapter_summary()))

    logger.info("Loaded MODEL: \n{}".format(llama.model))

    if CHAT_TEMPLATE:
        logger.info("****** RESULTS ARE GENERATED USING CHAT TEMPLATE ******")

    data = SumDataLoader(domain=test_domain, dataset_name=test_dataset_name, training_samples=training_samples,
                         eval_samples=eval_samples, test_samples=test_samples, sort_dataset_on_article_len=sort_data,
                         chat_template=CHAT_TEMPLATE)

    data.train_set = None
    data.validation_set = None

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, logger=logger,
                  col_name=all_adapters, data=data, chat_template=CHAT_TEMPLATE, metric_name=metric,
                  peft_full_name=all_adapters, device=device)
