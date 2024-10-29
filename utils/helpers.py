import torch
import yaml
from adapters import AutoAdapterModel
from transformers import AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # Meta-Llama-3-8B-Instruct


def get_pretrained_model(ah=True, quantization_config=None):
    ddtype = torch.bfloat16  # bfloat16
    compute_dtype = torch.bfloat16  # torch.bfloat16 if bf16 else torch.float32

    # if ah:
    #     print("Loading Model from Adapter Hub")
    #     model = AutoAdapterModel.from_pretrained(
    #         MODEL_ID,
    #         device_map="auto",
    #         quantization_config=quantization_config,
    #         torch_dtype=ddtype
    #     )
    #
    # else:
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


def convert_params_to_bfloat16(model, peft_name):
    for name, param in model.named_parameters():
        if peft_name in name:
            # logger.info("{} -> {}".format(name, param.dtype))
            param.data = param.data.to(torch.bfloat16)
            param.data = param.data.to(param.device)
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            # logger.info("To Dim1: {} -> {}".format(name, param.dtype))
            param.data = param.data.to(torch.float32)
            param.data = param.data.to(param.device)
    return model
