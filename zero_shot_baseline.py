import argparse

import torch

from dataset_lib import SumDataLoader
from testing_one_v_one_model import testing_model
from utils import LLaMAModelClass, torch_dtypes_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for Zero Shot Baseline")

    parser.add_argument("--domain", type=str, help="Domain name for dataset", choices=["scientific", "medical", "legal",
                                                                                       "news"], required=True)
    parser.add_argument("--dataset", type=str, help="Dataset to be used for training", required=True)
    parser.add_argument("--test_samples", type=int, default=1, help="Number of testing Samples")

    args = parser.parse_args()

    domain = args.domain
    dataset_name = args.dataset
    test_samples = args.test_samples
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    print("Running Zero Shot Baseline experiments for dataset {} in domain {} "
          "for {} test samples.".format(dataset_name, domain, test_samples))
    llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantize=False,
                            model_checkpoint=None, mlm=False, torch_dtype=torch_dtypes_dict["bf16"])

    data = SumDataLoader(domain=domain, dataset_name=dataset_name, training_samples=1, eval_samples=1,
                         test_samples=test_samples, sort_dataset_on_article_len=True)
    data.loading_dataset_splits()

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, data=data,
                  peft_full_name="llama_zero_shot_results", device=device)
