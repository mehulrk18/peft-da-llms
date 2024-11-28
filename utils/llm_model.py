import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLaMAModelClass:

    def __init__(self, hf_model_path: str = "", version: float = 3.0, instruct_mode: bool = False, model_checkpoint: str = None,
                 quantize: bool = False, mlm: bool = False, torch_dtype: torch.dtype = torch.bfloat16):
        self.version = float(version)
        self.instruct_mode = instruct_mode
        self.quantization_config = None
        self.mlm = mlm
        self.torch_dtype = torch_dtype
        self.model_path = hf_model_path

        if quantize:
            from transformers import BitsAndBytesConfig
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
        # MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # Meta-Llama-3-8B-Instruct

        if self.model_path is not None and self.model_path != "":
            self.model_id = self.model_path

        else:
            if self.version not in [2.0, 3.0, 3.1, 3.2]:
                raise ValueError("!! INVALID VERSION !!")

            model_dict = {
                "2.0": "Llama-2-7b-hf",
                "3.0": "Meta-Llama-3-8B",
                "3.1": "Llama-3.1-8B",
                "3.2": "Llama-3.2-3B"
            }

            if model_checkpoint is None or not model_dict:

                self.model_id = "meta-llama/"+model_dict[str(self.version)]

                if self.instruct_mode:
                    if self.version == 2.0:
                        print("Llama2 doesn't have an instruct model.")
                    else:
                        self.model_id = self.model_id + "-Instruct"
                        print("Loading Instruct Model!!")
            else:
                self.model_id = model_checkpoint

        print("Loading LLaMA's: {} Model for the process.".format(self.model_id))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.quantization_config,
            torch_dtype=torch_dtype
        )
        self.model.config.use_cache = False
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        self.model.config.torch_dtype = torch_dtype

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

    def __str__(self):
        return "LLaMA Model: {}".format(self.model_id)

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
