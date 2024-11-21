import argparse
import logging
import os

import torch
from dotenv import load_dotenv

import wandb

from dataset_lib import SumDataLoader
from testing_one_v_one_model import testing_model
from utils import LLaMAModelClass, torch_dtypes_dict, WandBLogger


def zero_shot_baseline(llama, domain, dataset_name, test_samples):
    run_name = 'zero_shot_learning_{}_{}_{}samples.log'.format(domain, dataset_name, test_samples)
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
    logger.info("Running Zero Shot Baseline experiments for dataset {} in domain {} for {} "
                "test samples.".format(dataset_name, domain, test_samples))

    data = SumDataLoader(domain=domain, dataset_name=dataset_name, training_samples=1, eval_samples=1,
                         test_samples=test_samples, sort_dataset_on_article_len=True)
    data.loading_dataset_splits()

    logger.info("About Dataset: \n{}".format(data.__str__()))

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data=data, logger=logger,
                  peft_full_name="zero_shot_{}_{}_results".format(domain, dataset_name), device=device,
                  chat_template=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for Zero Shot Baseline")

    # parser.add_argument("--domain", type=str, help="Domain name for dataset", choices=["scientific", "medical", "legal",
    #                                                                                    "news"], required=True)
    # parser.add_argument("--dataset", type=str, help="Dataset to be used for training", required=True)
    parser.add_argument("--test_samples", type=int, default=1, help="Number of testing Samples")

    args = parser.parse_args()

    # domain = args.domain
    # dataset_name = args.dataset
    test_samples = args.test_samples
    load_dotenv(".env")
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    # domains = ["news", "legal", "medical", "scientific"]
    # datasets = ["cnndm", "multilex", "pubmed", "arxiv"]
    domains_datasets = {
        "scientific": ["elsevier", "scitldr"], # arxiv
        "news": ["multinews", "xsum", "newsroom"], # cnndm
        "legal": ["eurlex", "billsum"], # multilex
        "medical": ["cord19", "mslr", "scilay"] # pubmed
    }
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantize=False,
                            model_checkpoint=None, mlm=False, torch_dtype=torch_dtypes_dict["bf16"])

    for domain, datasets in domains_datasets.items(): #zip(domains, datasets):
        for dataset_name in datasets:
            zero_shot_baseline(llama, domain, dataset_name, test_samples)
