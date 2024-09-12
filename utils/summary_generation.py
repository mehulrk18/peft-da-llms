import torch
from dataset_lib import inference_prompt


# TODO: Improve Generation and Evaluation.

def generate_summary(model, tokenizer, text, device):
    content = inference_prompt(text)
    # print("Text: \n", text)
    inputs = tokenizer(content, return_tensors="pt").to(device)
    in_len = len(inputs["input_ids"][0])
    with torch.inference_mode():
        summary_ids = model.generate(**inputs,
                                     # max_length=512, # do_sample=True,  # Enable sampling
                                     top_k=50,  # Top-k sampling
                                     num_return_sequences=1,  # Generate a single sequence
                                     # early_stopping=True,
                                     # temprature=0.001,
                                     max_new_tokens=150)
    # print("Truth:\n{}\n\n\nPrediction:\n{} ".format(truth, summary))
    summary = tokenizer.decode(summary_ids[0][in_len:], skip_special_tokens=True)

    return summary


def evaluate_summary(model, tokenizer, text, truth, metric, device):
    summary = generate_summary(model, tokenizer, text, device)

    print("Truth:\n{}\n\n\nPrediction:\n{} ".format(truth, summary))

    print("\n\n\nRouge Scores: ", metric.compute(references=[truth], predictions=[summary]))
    # return summary
