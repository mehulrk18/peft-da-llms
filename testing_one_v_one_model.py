import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer

from dataset_lib import inference_prompt, SumDataLoader
from utils import generate_summary, get_pretrained_model, MODEL_ID, rouge_metric


def testing_model(llama_model, llama_tokenizer, domain, peft_full_name, device):
    print("MODEL: \n", llama_model)

    # testing the model with Test data.
    def inference_prompt_processing(sample):
        if "sources" in sample.keys():
            sample["article"] = sample.pop("sources")

        text = [inference_prompt(article=article) for article in sample["article"]]
        return {
            "text": text
        }
    data = SumDataLoader(dataset_name=domain)

    random_text = """
            Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
        """.strip()

    summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_text, device=device)
    # summ = summarize(inputs=random_text, return_text=False)
    print("Summary of Random Text from Wikipedia: \n", summ)
    try:
        with open("random_text_{}_pipeline.txt".format(peft_full_name), "w") as f:
            f.write("Wikipedia Article: \n{} \n\n\n\n Summary: \n".format(random_text, summ))
            print("Written Random article summary")
    except:
        pass

    data.loading_dataset_splits()

    data.train_set = None
    data.validation_set = None

    data.test_set = data.test_set.map(inference_prompt_processing, batched=True)
    df_test_data = pd.DataFrame(data=data.test_set)

    # TODO: write the testing funciton with a metric.
    test_summaries = {
        "truth": [],
        "prediction": []
    }

    # for arxiv and pubmed

    for i in range(len(df_test_data)):
        summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=df_test_data["article"][i],
                                   device=device)
        test_summaries["truth"].append(df_test_data["abstract"][i])
        test_summaries["prediction"].append(summary)

    metric = rouge_metric()
    scores = metric.compute(predictions=test_summaries["prediction"], references=test_summaries["truth"])
    df_sum = pd.DataFrame(test_summaries)
    # print("Rouge Scores: ", scores)
    file_name = "Test_summaries_{}.csv".format(peft_full_name)
    df_sum.to_csv(file_name, index=False)

    print("\n\n\nSummaries with Rouge Score {} saved to file {}!!!!".format(scores, file_name))


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--trained_peft_path", type=str, help="Path of the PEFT to be loaded.")
    # parser.add_argument("--domain", type=str, help="Domain name for dataset")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of Samples to be tested")

    args = parser.parse_args()

    llama_model = get_pretrained_model()

    try:
        from google.colab import drive

        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        print("Exception: ", e)
        main_directory = ""

    # domain = args.domain
    trained_peft_path = main_directory + args.trained_peft_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_dir = trained_peft_path.split("/")[-1]
    domain, peft_name = tuple(peft_dir.split("_")[:2])
    loaded_peft = llama_model.load_adapter(trained_peft_path, with_head=True)
    peft_layer = "{}_{}".format(domain, peft_name)
    llama_model.set_active_adapters(loaded_peft)
    llama_model.adapter_to(loaded_peft, device=device)

    llama_model = llama_model.to(torch.bfloat16)

    llama_model.enable_input_require_grads()
    llama_model.gradient_checkpointing_enable()

    print("\nLLaMA Model's Summary:\n", llama_model.adapter_summary())

    # for name, param in llama_model.named_parameters():
    #     if "lora" in name:
    #         print(name, param.dtype)
    #         param.data = param.data.to(torch.bfloat16)
        # if param.ndim == 1:
        #     # cast the small parameters (e.g. layernorm) to fp32 for stability
        #     print(name, param.dtype)
            # param.data = param.data.to(torch.float32)

    llama_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        padding_side="right",
        tokenizer_type="llama",
        trust_remote_code=True,
        use_fast=True
    )

    testing_model(llama_model=llama_model, llama_tokenizer=llama_tokenizer, domain=domain, peft_full_name=peft_dir, device=device)
