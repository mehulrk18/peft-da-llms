import torch
from dataset_lib import llama3_testing_prompt, chat_template_prompt_inference


# TODO: Improve Generation and Evaluation.

def generate_summary(model, tokenizer, content, device, prompt, chat_template=False):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # print("Text: \n", text)
    if not chat_template:
        # content = inference_prompt(content)
        content = llama3_testing_prompt(content=content, system_prompt=prompt)
        inputs = tokenizer(content, return_tensors="pt").to(device)
        in_len = len(inputs["input_ids"][0])
        # with torch.inference_mode() and torch.cuda.amp.autocast():
        with torch.amp.autocast(device):  #torch.cuda.amp.autocast():
            if hasattr(model, "module"):
                summary_ids = model.module.generate(**inputs,
                                             # max_length=512,
                                             do_sample=True,  # Enable sampling
                                             top_k=50,  # Top-k sampling
                                             num_return_sequences=1,  # Generate a single sequence
                                             eos_token_id=terminators,
                                             # early_stopping=True,
                                             temperature=0.3,
                                             max_new_tokens=256) # 256
            else:
                summary_ids = model.generate(**inputs,
                                             # max_length=512,
                                             do_sample=True,  # Enable sampling
                                             top_k=50,  # Top-k sampling
                                             num_return_sequences=1,  # Generate a single sequence
                                             eos_token_id=terminators,
                                             # early_stopping=True,
                                             temperature=0.3,
                                             max_new_tokens=256)  # 256
        summary = tokenizer.decode(summary_ids[0][in_len:], skip_special_tokens=True)

    else:
        content = chat_template_prompt_inference(content=content, system_prompt=prompt)
        input_ids = tokenizer.apply_chat_template(
            conversation=content,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.amp.autocast(device):
            if hasattr(model, "module"):
                summary_ids = model.module.generate(
                    input_ids,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )
            else:
                summary_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )
        summary = tokenizer.decode(summary_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return summary


def evaluate_summary(model, tokenizer, text, truth, metric, device):
    summary = generate_summary(model, tokenizer, text, device)

    print("Truth:\n{}\n\n\nPrediction:\n{} ".format(truth, summary))

    print("\n\n\nRouge Scores: ", metric.compute(references=[truth], predictions=[summary]))
    # return summary
