import argparse
import logging
import os
import statistics
import json

import adapters
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv

from dataset_lib import SumDataLoader, DEFAULT_DOMAIN_PROMPT, DEFAULT_SYSTEM_PROMPT, SumDomains, datasets_info_dict
from utils import generate_summary, rouge_metric, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, torch_dtypes_dict, WandBLogger, check_and_return_df, bertscore_metric, \
    bleu_metric, bleurt_metric, read_yaml, meteor_metric, power_set


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--config_pefts_file_path", default=None, type=str, help="Path of the config file containing pefts and dataset for multiple pefts inference")
    parser.add_argument("--peft_path", type=str, default=None, help="For single peft")
    parser.add_argument("--peft_dir", type=str, default="trained_pefts/", help="Storage directory")
    parser.add_argument("--test_dataset", default=None, type=str, help="Only test dataest.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"],
                        help="Torch Data Type to be used")
    parser.add_argument("--metric", type=str, required=True, choices=["rouge", "bertscore", "bleu", "bleurt", "all"],
                        help="Metric to be used for testing, pass 'all' if you want test on all")
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize the model")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    parser.add_argument("--mlm", type=bool, default=False, help="Using attention mask")

    main_directory, config_file = "", ""
    configs = {}

    args = parser.parse_args()
    if args.config_pefts_file_path is None and args.peft_path is None:
        raise ValueError("Please provide the path of the config file containing pefts and dataset for multiple pefts inference or the path of the peft for single peft inference")
    if args.config_pefts_file_path is not None:
        config_file = main_directory + args.config_pefts_file_path
        configs = read_yaml(file_name=config_file)
    elif args.peft_path is not None:
        configs["pefts"] = [args.peft_path]
    mlm = True if args.mlm else False
    chat_template = args.chat_template
    quantize = args.quantize
    metric = args.metric
    peft_dir = args.peft_dir
    torch_dtype = torch_dtypes_dict[args.torch_dtype]
    
    use_instruct_model = True # if "instruct" in trained_peft_path or args.chat_template else False
    # provider = "hf" if "hf" in trained_peft_path else "ah"
    
    dataset_name = args.test_dataset if args.test_dataset else configs["test_dataset"]

    power_sets = power_set(configs["pefts"])

    print("Power Sets: ", power_sets)

    for i, peft_set in enumerate(power_sets):
        if len(peft_set) < 2:
            continue
        configs["pefts"] = list(peft_set)


        # elif trained_peft_path.split("/")[0] ==  :# "saved_models":
        peft_names = []
        # if not zero_shot:
        zero_shot = False
        for i, path in enumerate(configs["pefts"]):
            a_name = path.split("_checkpoint")[0]
            configs["pefts"][i] = configs["pefts"][i] + "/" + a_name
            peft_names.append(a_name)

        provider = "hf"

        load_dotenv(".env")
        hf_token = os.getenv("HF_TOKEN")
        wandb_api_key = os.getenv("WANDB_API_KEY")
        run_name, wnb_run, logger, console_handler = None, None, None, None
        # run_name = 'unseen_data_inference_{}_domain_{}_{}.log'.format("cross" if "cross" in config_file else "within",
        #                                                               ("-".join(peft_names) if len(peft_names) > 1 else
        #                                                                peft_names[0]) if not zero_shot else "zero_shot",
        #                                                               dataset_name)

        run_name = 'holdout_data_inference_{}_domain_{}_{}.log'.format("cross" if "cross" in config_file else "within",
                                                                      ("-".join(peft_names) if len(peft_names) > 1 else
                                                                       peft_names[0]) if not zero_shot else "zero_shot",
                                                                      dataset_name)

        wnb_run = wandb.init(name=run_name)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
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

        logger.info("Configs file path: {}".format(config_file))
        logger.info("Device in use: {}".format(device))
        llama = None
        # llama_model = get_pretrained_model(ah=ah)
        llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize, mlm=mlm, torch_dtype=torch_dtype)
        # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

        # logger.info("Check point MODEL: \n{}".format(llama.model))

        # Method 1 - HuggingFace
        logger.info(" -->{}".format(configs))
        if not zero_shot:
            for a_path, a_name in zip(configs["pefts"], peft_names):
                llama.model.load_adapter(peft_dir+a_path, adapter_name=a_name)
        # llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
        # llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name=adapter_name,
        #                                                           torch_dtype=torch_dtype)
            logger.info("Active Adapters in Model before enabling adapters: {}".format(llama.model.active_adapters()))
            llama.model.set_adapter(peft_names)
            llama.model.enable_adapters()
            logger.info("Active Adapters in Model after enabling adapters: {}".format(llama.model.active_adapters()))

            # llama.model.set_adapter([peft_names])
            # logger.info("Active Adapters in Model: {}".format(llama.model.active_adapters()))
            llama.model = llama.model.to(torch_dtype)

        logger.info("Loaded MODEL: \n{}".format(llama.model))

        if chat_template:
            logger.info("****** RESULTS ARE GENERATED USING CHAT TEMPLATE ******")

        # data = SumDataLoader(domain=domain, dataset_name=dataset_name, training_samples=training_samples,
        #                      eval_samples=eval_samples, test_samples=test_samples, sort_dataset_on_article_len=sort_data,
        #                      chat_template=chat_template)

        domain = SumDomains("unseen_test")
        data_class = datasets_info_dict[domain][dataset_name]
        # data.loading_dataset_splits()


        # data.train_set = None
        # data.validation_set = None
        from inference_unseen_data import unseen_test_data_inference
        #TODO: do better nomenclature for the peft_full_name and col_name
        unseen_test_data_inference(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data_class=data_class,
                                   # peft_full_name=("multiple_"+"-".join(peft_names)+f"_{dataset_name}" if
                                   #                 len(peft_names) > 1 else peft_names[0]+f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   peft_full_name=("within_domain_test_" + "-".join(peft_names) + f"_{dataset_name}" if
                                                   len(peft_names) > 1 else peft_names[
                                                                                0] + f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   # col_name=("multiple_"+"-".join(peft_names)+f"_{dataset_name}" if
                                   #           len(peft_names) > 1 else peft_names[0]+f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   col_name=("within_domain_test_" + "-".join(peft_names) + f"_{dataset_name}" if
                                             len(peft_names) > 1 else peft_names[
                                                                          0] + f"_{dataset_name}") if not zero_shot else "zero_shot",
                                   logger=logger, device=device, chat_template=chat_template, metric_name=metric)
        wnb_run.finish()
