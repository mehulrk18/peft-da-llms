import logging
import torch
import yaml

MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # Meta-Llama-3-8B-Instruct


class WandBLogger(logging.StreamHandler):
    wandb: any
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    def emit(self, record):
        log_entry = self.format(record)
        # Log to WandB
        self.wandb.log({"log": log_entry})


torch_dtypes_dict = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}


def read_yaml(file_name: str):
    with open(file_name, "r") as yml:
        config = yaml.safe_load(yml)

    return config


def convert_model_adapter_params_to_torch_dtype(model, peft_name, torch_dtype):
    for name, param in model.named_parameters():
        if peft_name in name:
            # logger.info("{} -> {}".format(name, param.dtype))
            param.data = param.data.to(torch_dtype)
            param.data = param.data.to(param.device)
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            # logger.info("To Dim1: {} -> {}".format(name, param.dtype))
            param.data = param.data.to(torch_dtype)  # torch.float32)
            param.data = param.data.to(param.device)
    return model


def print_model_weights(model):
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            print(f"Layer: {name}, dtype: {layer.weight.dtype}")
        elif hasattr(layer, 'bias') and layer.bias is not None:
            print(f"Layer: {name}, dtype: {layer.bias.dtype}")
        # else:
        #     print(f"Layer: {name}, dtype: Not applicable (no weights)")
