import argparse
import logging
import os

import torch
from dotenv import load_dotenv

import wandb

from dataset_lib import SumDataLoader
from testing_one_v_one_model import testing_model
from utils import LLaMAModelClass, torch_dtypes_dict, WandBLogger


def hf_model_testing(llama, domain, dataset_name, test_samples, instruct, hf_model_id):
    hf_model_id = hf_model_id.replace("/", "-")
    run_name = '{}_{}{}_{}_{}samples.log'.format(hf_model_id, "instruct_" if instruct else "", domain, dataset_name, test_samples)
    logging.basicConfig(
        filename='logs/{}'.format(run_name),
        # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    wnb_run = wandb.init(name=run_name)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    logger = logging.getLogger()
    wnb = WandBLogger()
    wnb.wandb = wandb

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the log level for the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(wnb)
    logger.info("Running {} experiments for dataset {} in domain {} for {} "
                "test samples.".format(hf_model_id, dataset_name, domain, test_samples))

    data = SumDataLoader(domain=domain, dataset_name=dataset_name, training_samples=1, eval_samples=1,
                         test_samples=test_samples, sort_dataset_on_article_len=True)
    data.loading_dataset_splits()

    logger.info("About Dataset: \n{}".format(data.__str__()))

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data=data, logger=logger,
                  peft_full_name="{}_{}_{}_results".format(hf_model_id, domain, dataset_name), device=device,
                  chat_template=instruct, col_name=hf_model_id+"_instruct" if instruct else hf_model_id)
    wnb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for Zero Shot Baseline")

    parser.add_argument("--hf_model_id", type=str, help="Give HF Model Path", required=True)
    parser.add_argument("--domain", type=str, help="Domain name for dataset", choices=["scientific", "medical", "legal",
                                                                                       "news"], required=True)
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--test_samples", type=int, default=1, help="Number of testing Samples")
    parser.add_argument("--instruct", type=bool, default=False, help="Use Instruct Model")

    args = parser.parse_args()

    domain = args.domain
    dataset_name = args.dataset_name
    hf_model_id = args.hf_model_id
    test_samples = args.test_samples
    instruct = True if args.instruct else False
    load_dotenv(".env")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    llama = LLaMAModelClass(hf_model_path=hf_model_id, version=3.0, instruct_mode=instruct, quantize=False,
                            model_checkpoint=None, mlm=instruct, torch_dtype=torch_dtypes_dict["bf16"])

    try:
        if llama.tokenizer.chat_template is None:
            instruct = False
    except Exception as e:
        print("Error in chat_template: ", e)
        instruct = False

    # for domain, datasets in domains_datasets.items(): #zip(domains, datasets):
    #     for dataset_name in datasets:
    print("** Generating results from {} in domain {} for dataset with {} samples with "
          "domain specific prompts".format(hf_model_id, dataset_name, domain, test_samples))
    hf_model_testing(llama, domain, dataset_name, test_samples, instruct, hf_model_id)
