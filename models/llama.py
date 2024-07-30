from langchain import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch

from da_llms.peft.representation_fine_tuning import reft


def llama_model(bnb=False):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
            quantization_config=bnb_config
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )

    model.tie_weights()

    # calling a the PEFT Method
    model = reft(model)

    generation_config = GenerationConfig.from_pretrained(model_id)

    # pipe = pipeline(
    #     "text-summarization", # "text-generation"
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_length=2048,
    #     temperature=0,
    #     top_p=0.95,
    #     repetition_penalty=1.15,
    #     generation_config=generation_config,
    # )

    return tokenizer, model
    # local_llm = HuggingFacePipeline(pipeline=pipe)
    # print("LLaMA loaded")
    #
    # return local_llm

