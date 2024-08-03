import torch
from huggingface_hub import login
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, get_peft_model, PeftConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import evaluate

bits = 4
compute_dtype = torch.float32
load_adapter_path = "saved_models/arxiv_lora_adapter"

login(token="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")

bnb_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

peft_config = PeftConfig.from_pretrained(load_adapter_path)
model_id = peft_config.base_model_name_or_path


loaded_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32, #bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", tokenizer_type="llama",
                                          trust_remote_code=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
loaded_model.resize_token_embeddings(len(tokenizer))
loaded_model.config.torch_dtype = torch.float32

loaded_model = prepare_model_for_kbit_training(loaded_model)
loaded_model = get_peft_model(loaded_model, peft_config)
loaded_model.load_adapter(load_adapter_path, "arxiv_lora_adapter")


rouge = evaluate.load("rouge")

def summarization_model(model, text, true_summary=""):
    max_seq_len = 512
    inputs = tokenizer(text, return_tensors="pt", max_length=max_seq_len, truncation=True, padding="max_length")
    inputs = inputs.to(model.device)
    # model.eval()
    # outputs = model.generate(inputs.input_ids, max_length=1024, num_beams=5, early_stopping=True)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=min(len(true_summary), 1024),
        num_beams=5,
        early_stopping=True
    )

    pred_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # for s1, s2 in zip(sums, summaries):
    if len(true_summary) > 0:
        rouge_score = rouge.compute(predictions=[pred_summary], references=[true_summary])

        print("Rouge: ", rouge_score)

        # print("actual: ", true_summary)
    # print("\n\npred: ", pred_summary)


dataset = load_from_disk("domains/arxiv_summarization")
dataset = dataset["test"]

print("Device: ", next(loaded_model.parameters()).device)
print(dataset[4].keys()) #gen_art[4]["summary"])
print("************(************(************(************(************")
summarization_model(loaded_model, dataset[4]["article"],dataset[4]["abstract"])