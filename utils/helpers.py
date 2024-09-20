import torch
import yaml
from adapters import AutoAdapterModel
from transformers import AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B" # Meta-Llama-3-8B-Instruct
# llama31 = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # works only with transformers==4.43.3


def get_pretrained_model(fine_tuning=True, quantization_config=None):
    ddtype = torch.bfloat16  # bfloat16
    compute_dtype = torch.bfloat16  # torch.bfloat16 if bf16 else torch.float32

    if fine_tuning:
        print("Loading Model from Adapter Hub")
        model = AutoAdapterModel.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=ddtype
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=ddtype
        )

    model.config.use_cache = False
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = compute_dtype
    # (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float16))

    return model


def read_yaml(file_name: str):
    with open(file_name, "r") as yml:
        config = yaml.safe_load(yml)

    return config
