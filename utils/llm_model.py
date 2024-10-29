import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLaMAModelClass():

    def __init__(self, version: float = 3.0, instruct_mode: bool = False, model_checkpoint: str = None,
                 quantization_config=None, mlm: bool = False):
        self.version = float(version)
        self.instruct_mode = instruct_mode
        self.quantization_config = quantization_config
        self.mlm = mlm
        # MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # Meta-Llama-3-8B-Instruct

        if self.version not in [2.0, 3.0, 3.1, 3.2]:
            raise ValueError("!! INVALID VERSION !!")

        model_dict = {
            "2.0": "Llama-2-7b-hf",
            "3.0": "Meta-Llama-3-8B",
            "3.1": "Llama-3.1-8B",
            "3.2": "Llama-3.2-8B"
        }

        if model_checkpoint is None or not model_dict:

            self.model_id = "meta-llama/"+model_dict[str(self.version)]

            if self.instruct_mode:
                if self.version == 2.0:
                    print("Llama2 doesn't have an instruct model.")
                else:
                    self.model_id = self.model_id + "-Instruct"

        else:
            self.model_id = model_checkpoint

        print("Loading LLaMA's: {} Model for the process.".format(self.model_id))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )
        self.model.config.use_cache = False
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        self.model.config.torch_dtype = torch.bfloat16

        print("*** Model Loaded ***")

        print("Loading LLaMA's: {} Tokenizer for the process.".format(self.model_id))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="right",
            tokenizer_type="llama",
            trust_remote_code=True,
            use_fast=True
        )
        self.tokenizer.add_special_tokens({
            "eos_token": self.tokenizer.convert_ids_to_tokens(self.model.config.eos_token_id),
            "bos_token": self.tokenizer.convert_ids_to_tokens(self.model.config.bos_token_id),
        })
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.mlm:
            print("*** Adding the Mask to the LM Tokenizer ***")
            self.tokenizer.add_special_tokens({
                "mask_token": "<|mask|>"
            })
            # self.tokenizer.mask_token_id = -100
            self.model.resize_token_embeddings(len(self.tokenizer))

        print("*** Tokenizer Loaded ***")

    def return_model(self):
        return self.model

    def reassign_model(self, model):
        self.model = None
        self.model = model

    def return_tokenizer(self):
        return self.tokenizer

    def return_model_id(self):
        return self.model_id

    def __call__(self):
        return self.return_model(), self.return_tokenizer()
