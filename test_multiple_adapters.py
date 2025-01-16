import argparse
import logging
import os

import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from dataset_lib import SumDataLoader
from testing_one_v_one_model import testing_model
from utils import generate_summary, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, read_yaml, torch_dtypes_dict, WandBLogger

global CHAT_TEMPLATE


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    global CHAT_TEMPLATE

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--trained_adapter_config_file", type=str, required=True, help="yaml file paths to the of trained adapters")

    parser.add_argument("--peft_dir", type=str, default="trained_pefts/", help="Storage directory")
    parser.add_argument("--test_domain", required=True, type=str, help="Testing domain.")
    parser.add_argument("--test_dataset", required=True, type=str, help="Only test dataest.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Torch Data Type to be used")
    parser.add_argument("--metric", type=str, required=True, choices=["rouge", "bertscore", "bleu", "bleurt", "all"],
                        help="Metric to be used for testing, pass 'all' if you want test on all")
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize the model")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    parser.add_argument("--mlm", type=bool, default=False, help="Using attention mask")
    parser.add_argument("--sorted_dataset", type=bool, default=False, help="Sorted dataset")
    parser.add_argument("--test_samples", type=int, default=150, help="Test samples")
    

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        main_directory = ""

    args = parser.parse_args()
    trained_peft_config_path = main_directory + args.trained_adapter_config_file
    testing_configs = read_yaml(file_name=trained_peft_config_path)
    # mlm = testing_configs["mlm"]
    # training_samples = testing_configs["training_samples"]
    # eval_samples = testing_configs["eval_samples"]
    # test_samples = testing_configs["test_samples"]
    # CHAT_TEMPLATE = testing_configs["chat_template"]
    # use_instruct_model = testing_configs["instruct_mode"]
    provider = "hf" #testing_configs["provider"]
    # metric = testing_configs["metric"]
    # torch_dtype = torch_dtypes_dict[testing_configs["torch_dtype"]]
    # quantize = testing_configs.get("torch_dtype", False)

    mlm = True if args.mlm else False
    CHAT_TEMPLATE = args.chat_template
    quantize = args.quantize
    metric = args.metric
    peft_dir = args.peft_dir
    samples = args.test_samples
    sort_data = args.sorted_dataset
    test_domain = args.test_domain
    test_dataset = args.test_dataset
    torch_dtype = torch_dtypes_dict[args.torch_dtype]
    
    use_instruct_model = True # if "instruct" in trained_peft_path or args.chat_template else False
    # provider = "hf" if "hf" in trained_peft_path else "ah"
    
    # dataset_name = args.test_dataset # configs["dataset_name"]

    # adapter_paths = testing_configs["trained_adapters_path"]

    adapter_names = []

    # for path in adapter_paths:
    #     a_name = path.split("/")[-1]
    #     adapter_names.append(a_name)
    
    for i, path in enumerate(testing_configs["pefts"]):
        a_name = path.split("_checkpoint")[0]
        testing_configs["pefts"][i] = testing_configs["pefts"][i]+ "/"+ a_name
        adapter_names.append(a_name)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    all_adapters = "-".join(adapter_names)
    load_dotenv(".env")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    run_name = 'testing_multiple_pefts_{}_data_{}-{}_{}samples.log'.format(all_adapters, test_domain, test_dataset, samples)
    wnb_run = wandb.init(name=run_name)
    logging.basicConfig(
        filename=main_directory + 'logs/{}'.format(run_name),  # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    logger = logging.getLogger()
    wnb = WandBLogger()
    wnb.wandb = wandb

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the log level for the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(wnb)

    logger.info("Device in use: {}".format(device))

    logger.info("Adapters:-> {}".format(testing_configs["pefts"]))
    logger.info("On Dataset :-> {}_{}".format(test_domain, test_dataset))

    # llama_model = get_pretrained_model(ah=ah)
    llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize, mlm=mlm,
                            torch_dtype=torch_dtype)
    # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

    logger.info("Check point MODEL: \n{}".format(llama.model))

    # if provider == "hf":
    # Method 1 - HuggingFace
    from peft import PeftModel
    for a_path, a_name in zip(testing_configs["pefts"], adapter_names): # zip(adapter_paths, adapter_names):
        llama.model.load_adapter(peft_dir+a_path, adapter_name=a_name)

    # for i in range(len(adapter_names)):
    #     adapter_names[i] = "trained_" + adapter_names[i]
    # llama.model.load_adapter(lora_path, adapter_name="m_lora")
    # llama.model = PeftModel.from_pretrained(llama.model, trained_peft_path, adapter_name=adapter_name) #, use_safetensors=True)
    # # llama.model = llama.model.merge_and_unload()
    # llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
    logger.info("Active Adapters in Model before enabling adapters: {}".format(llama.model.active_adapters()))
    llama.model.set_adapter(adapter_names)
    llama.model.enable_adapters()
    logger.info("Active Adapters in Model after enabling adapters: {}".format(llama.model.active_adapters()))
    # TODO: check if below commented line is required
    # llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name="trained_", torch_dtype=torch_dtype)
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

    # if os.path.exists("summaries/summaries_{}_{}_{}samples.csv".format(test_domain, test_dataset, samples))

    data = SumDataLoader(domain=test_domain, dataset_name=test_dataset, training_samples=1,
                         eval_samples=1, test_samples=samples, sort_dataset_on_article_len=sort_data,
                         chat_template=CHAT_TEMPLATE)

    data.train_set = None
    data.validation_set = None

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, logger=logger,
                  col_name="multiple_"+all_adapters, data=data, chat_template=CHAT_TEMPLATE, metric_name=metric,
                  peft_full_name="multiple_"+all_adapters, device=device)

    wnb_run.finish()
