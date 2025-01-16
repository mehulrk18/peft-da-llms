import argparse
import json
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime
from typing import List, Optional

# was originally in the metrics/

root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Adjust path as needed

print(root_dir)
sys.path.append(root_dir)

import nltk
import numpy as np

# from metrics.FActScore.factscore.factscorer import FactScorer
from FActScore.factscore.factscorer import FactScorer

nltk.download("punkt_tab")
from dotenv import load_dotenv

load_dotenv()


class GenFact:
    def __init__(self, args: Optional[List[str]] = None):
        # print(args)
        self.args = parse_options(sys.argv[1:] if args is None else args)
        # print(self.args)

        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.ERROR
            if self.args.print_rate_limit_error
            else logging.CRITICAL,
        )

        self.log_dir = self.create_log_folder()
        self.fs = FactScorer(
            model_name=self.args.model_name,
            data_dir=self.args.data_dir,
            model_dir=self.args.model_dir,
            cache_dir=self.args.cache_dir,
            openai_key=self.args.openai_key,
            cost_estimate=self.args.cost_estimate,
            abstain_detection_type=self.args.abstain_detection_type,
            grounding_provided=self.args.grounding_provided,
        )

    def run_factscrorer(self, grounding_provided: bool) -> dict:
        tot = 0
        topics, generations, atomic_facts, groundings = [], [], [], []
        with open(self.args.input_path) as f:
            for line in f:
                dp = json.loads(line)
                tot += 1
                if self.args.use_atomic_facts:
                    assert (
                        "annotations" in dp
                    ), "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                    if dp["annotations"] is None:
                        continue
                    topics.append(dp["topic"])
                    generations.append(dp["output"])
                    atomic_facts.append(
                        [
                            atom["text"]
                            for sent in dp["annotations"]
                            for atom in sent["model-atomic-facts"]
                        ]
                    )
                else:
                    topics.append(dp["topic"])
                    generations.append(dp["output"])
                if self.args.grounding_provided:
                    groundings.append(dp["input"])

                if self.args.n_samples is not None and tot == self.args.n_samples:
                    break
        out = self.fs.get_score(
            topics=topics,
            generations=generations,
            groundings=groundings,
            gamma=self.args.gamma,
            atomic_facts=atomic_facts if self.args.use_atomic_facts else None,
            knowledge_source=self.args.knowledge_source,
            verbose=self.args.verbose,
            grounding_provided=grounding_provided,
        )
        print("Using intrinsic Fact Checking")
        logging.critical("FActScore = %.1f%%" % (100 * out["score"]))
        if "init_score" in out:
            logging.critical(
                "FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"])
            )
        logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        logging.critical(
            "# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"])
        )

        # Save out as a json file
        with open(
            self.args.input_path.replace(".jsonl", "_factscore_output.json"), "w"
        ) as f:
            f.write(json.dumps(out) + "\n")

        self.factscore_logs = {
            "score": out["score"],
            "topics": topics,
            "decisions": out["decisions"],
            "wrong_facts": out["wrong_facts"],
            # "groundings": groundings,
            # "generations": generations,
            "num_atomic_facts": out["num_facts_per_response"],
            "grounding_provided": grounding_provided,
        }

        return self.factscore_logs

    def write_logs(self, out: json, fname: str):
        fname = os.path.join(self.log_dir, fname)
        with open(fname, "w") as fp:
            json.dump(out, fp)

    def fs_get_extrinsic_af(
        self, topics, wrong_facts, groundings, generations, grounding_provided
    ):
        # Check if the wrongly classified facts are "wrong" or just not present in the article.
        extrinsic_af = self.fs.get_extrinsic_af(
            topics=topics,
            wrong_facts=wrong_facts,
            groundings=groundings,
            generations=generations,
            grounding_provided=grounding_provided,
        )

        return extrinsic_af

    def fs_extrinsic_score(self, fs_extrinsic_af: dict):
        extrinsic_out = self.fs.get_extrinsic_score(
            topics=self.factscore_logs["topics"],
            extrinsic_facts=fs_extrinsic_af["extrinsic_facts"],
            generations=self.factscore_logs["generations"],
            verbose=False,
            grounding_provided=False,
        )
        return extrinsic_out

    def create_log_folder(
        self,
    ):
        date_time = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.now())
        run_name = os.path.basename(self.args.input_path).replace(".jsonl", "")

        folder = os.path.join("results", "genfact", run_name, date_time)
        os.makedirs(folder, exist_ok=True)
        print(f"Run outputs would be locally stored at {folder}")
        return folder

    def get_updated_score(self, factscore_out, fs_extrinsic_af) -> float:
        decision_before = factscore_out["decisions"]
        decision_after = fs_extrinsic_af["decisions"]
        count = 0

        for idx, afs in enumerate(decision_after):
            if len(afs) > 0:
                for af in afs:
                    if (
                        decision_before[idx][af["idx"]]["is_supported"]
                        != af["is_supported"]
                    ):
                        print(
                            f"Updating the decision for the Atomic Fact: {af} for sample {idx}"
                        )
                        decision_before[idx][af["idx"]]["is_supported"] = af[
                            "is_supported"
                        ]
                        count += 1
        scores = [
            np.mean([d["is_supported"] for d in decisions])
            for decisions in decision_before
        ]
        hallucinations = [
            [d for d in decisions if not d["is_supported"]]
            for decisions in decision_before
        ]

        updated_score = np.mean(scores)
        logging.critical(
            "FActScore After extrinsic check = %.1f%%" % (100 * updated_score)
        )
        logging.critical(
            f"Updated decision on {str(count)} Facts after running Extrinsic check"
        )
        # if "init_score" in extrinsic_out:
        #    logging.critical("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
        # logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        # logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
        return updated_score, hallucinations


def parse_options(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="llm.json")
    parser.add_argument("--input_path", type=str, default=args.input_path)
    parser.add_argument("--model_name", type=str, default="GPT-4o-mini")
    parser.add_argument(
        "--gamma", type=int, default=10, help="hyperparameter for length penalty"
    )

    parser.add_argument("--openai_key", type=str, default=args.openai_key)
    # @todo: Fix these paths to be defined from the majorityvoter or command line and revert this to the original factscore code stucture
    parser.add_argument(
        "--data_dir",
        type=str,
        # default=os.path.join(os.getcwd(), "metrics/FActScore/.cache/factscore"),
        default=os.path.join(os.getcwd(), "FActScore/.cache/factscore"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.getcwd(), "FActScore/.cache/factscore"),
        # default=os.path.join(os.getcwd(), "metrics/FActScore/.cache/factscore"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(os.getcwd(), "FActScore/.cache/factscore"),
        # default=os.path.join(os.getcwd(), "metrics/FActScore/.cache/factscore"),
    )
    parser.add_argument("--knowledge_source", type=str, default=None)

    parser.add_argument(
        "--cost_estimate",
        type=str,
        default="consider_cache",
        choices=["consider_cache", "ignore_cache"],
    )
    parser.add_argument(
        "--abstain_detection_type",
        type=str,
        default=None,
        choices=["perplexity_ai", "generic", "none"],
    )
    parser.add_argument("--use_atomic_facts", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="for printing out the progress bar"
    )
    parser.add_argument(
        "--print_rate_limit_error",
        action="store_true",
        help="for printing out rate limit error when using OpenAI keys",
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument(
        "--grounding_provided", type=bool, default=args.grounding_provided
    )

    args = parser.parse_args()
    return args


def calc_factscore(arg_dict):
    args = Namespace(**arg_dict)
    print(args.input_path)

    genFact = GenFact(args)

    print("Running Factscore with grounded document")
    factscore_out = genFact.run_factscrorer(grounding_provided=args.grounding_provided)
    genFact.write_logs(factscore_out, fname="factscore_grounded.json")

    return {"score": factscore_out["score"], "num_atomic_facts": factscore_out["num_atomic_facts"]}


# if __name__ == "__main__":
#     # calc_factscore(
#     #     input_path="results/summarization_csv",
#     #     grounding_provided=True,
#     # )
