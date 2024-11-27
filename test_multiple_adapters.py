import argparse
import logging

import pandas as pd
import torch

from dataset_lib import SumDataLoader
from utils import generate_summary, rouge_metric, LLaMAModelClass, \
    convert_model_adapter_params_to_torch_dtype, read_yaml, torch_dtypes_dict


global CHAT_TEMPLATE


def testing_model(llama_model, llama_tokenizer, test_samples, peft_full_name, device):
    # testing the model with Test data.
    def inference_prompt_processing(sample):
        if "sources" in sample.keys():
            sample["article"] = sample.pop("sources")

        if CHAT_TEMPLATE:
            from dataset_lib import chat_template_prompt_inference
            text = [chat_template_prompt_inference(article=article) for article in sample["article"]]

            return {
                "text": text
            }
        else:
            # text = [inference_prompt(article=article) for article in sample["article"]]
            from dataset_lib import llama3_testing_prompt
            text = [llama3_testing_prompt(article=article) for article in sample["article"]]

            return {
                "text": text
            }

    random_text = """
            Rome had begun expanding shortly after the founding of the Republic in the 6th century BC, though it did not expand outside the Italian Peninsula until the 3rd century BC, during the Punic Wars, afterwhich the Republic expanded across the Mediterranean.[5][6][7][8] Civil war engulfed Rome in the mid-1st century BC, first between Julius Caesar and Pompey, and finally between Octavian (Caesar's grand-nephew) and Mark Antony. Antony was defeated at the Battle of Actium in 31 BC, leading to the annexation of Egypt. In 27 BC, the Senate gave Octavian the titles of Augustus ("venerated") and Princeps ("foremost"), thus beginning the Principate, the first epoch of Roman imperial history. Augustus' name was inherited by his successors, as well as his title of Imperator ("commander"), from which the term "emperor" is derived. Early emperors avoided any association with the ancient kings of Rome, instead presenting themselves as leaders of the Republic.\nThe success of Augustus in establishing principles of dynastic succession was limited by his outliving a number of talented potential heirs; the Julio-Claudian dynasty lasted for four more emperors—Tiberius, Caligula, Claudius, and Nero—before it yielded in AD 69 to the strife-torn Year of the Four Emperors, from which Vespasian emerged as victor. Vespasian became the founder of the brief Flavian dynasty, to be followed by the Nerva–Antonine dynasty which produced the "Five Good Emperors": Nerva, Trajan, Hadrian, Antoninus Pius and the philosophically inclined Marcus Aurelius. In the view of the Greek historian Cassius Dio, a contemporary observer, the accession of the emperor Commodus in AD 180 marked the descent "from a kingdom of gold to one of rust and iron"[9]—a famous comment which has led some historians, notably Edward Gibbon, to take Commodus' reign as the beginning of the decline of the Roman Empire.
        """.strip()

    summ = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=random_text, device=device, chat_template=CHAT_TEMPLATE)
    # summ = summarize(inputs=random_text, return_text=False)
    logger.info("Summary of Random Text from Wikipedia: \n{}".format(summ))
    try:
        with open("random_ip_summaries/random_text_{}.txt".format(peft_full_name), "w") as f:
            f.write("Wikipedia Article: \n{} \n\n\n\n Summary:{}\n".format(random_text, summ))
            logger.info("Written Random article summary")
    except Exception as e:
        logger.error("Exception: ".format(e))
        pass

    data.test_set = data.test_set.map(inference_prompt_processing, batched=True)
    df_test_data = pd.DataFrame(data=data.test_set)

    # TODO: write the testing funciton with a metric.
    test_summaries = {
        "truth": [],
        "prediction": []
    }

    # for arxiv and pubmed
    min_samples = min(test_samples, len(df_test_data))
    for i in range(min_samples):
        logger.info("Summary for {} sample".format(i))
        summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=df_test_data["article"][i],
                                   device=device, chat_template=CHAT_TEMPLATE)
        test_summaries["truth"].append(df_test_data["abstract"][i])
        test_summaries["prediction"].append(summary)
        del summary

    metric = rouge_metric()
    scores = metric.compute(predictions=test_summaries["prediction"], references=test_summaries["truth"])
    df_sum = pd.DataFrame(test_summaries)
    # logger.info("Rouge Scores: ", scores)
    file_name = "Test_summaries_{}_{}samples.csv".format(peft_full_name, min_samples)
    df_sum.to_csv(file_name, index=False)

    logger.info("\n\n\nSummaries with Rouge Score {} saved to file {}!!!!".format(scores, file_name))


if __name__ == "__main__":

    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    global CHAT_TEMPLATE

    parser = argparse.ArgumentParser(description="Argument parser to fetch PEFT and Dataset (domain) for training")

    parser.add_argument("--trained_adapter_config_file", type=str, required=True,
                        help="yaml file containing configs and paths of trained adapters.")

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        main_directory = ""

    args = parser.parse_args()
    trained_peft_config_path = main_directory + args.trained_adapter_config_file
    testing_configs = read_yaml(file_name=trained_peft_config_path)
    mlm = testing_configs["mlm"]
    training_samples = testing_configs["training_samples"]
    eval_samples = testing_configs["eval_samples"]
    test_samples = testing_configs["test_samples"]
    sort_data = testing_configs["sorted_dataset"]
    CHAT_TEMPLATE = testing_configs["chat_template"]
    use_instruct_model = testing_configs["instruct_mode"]
    provider = testing_configs["provider"]
    domain = testing_configs["domain"]
    torch_dtype = torch_dtypes_dict[testing_configs["torch_dtype"]]
    quantize = testing_configs.get("torch_dtype", False)

    adapter_paths = testing_configs["trained_adapters_path"]

    adapter_names = []

    for path in adapter_paths:
        a_name = path.split("/")[-1]
        adapter_names.append(a_name)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available else "cpu")
    all_adapters = "-".join(adapter_names)
    logging.basicConfig(
        filename=main_directory + 'logs/testing_{}_{}samples.log'.format(all_adapters, test_samples),  # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    logger = logging.getLogger()
    logger.info("Device in use: {}".format(device))

    # llama_model = get_pretrained_model(ah=ah)
    llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantize=quantize, mlm=mlm,
                            torch_dtype=torch_dtype)
    # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

    logger.info("Check point MODEL: \n{}".format(llama.model))

    if provider == "hf":
        # Method 1 - HuggingFace
        from peft import PeftModel
        for a_path, a_name in zip(adapter_paths, adapter_names):
            llama.model.load_adapter(a_path, adapter_name="trained_"+a_name)

        for i in range(len(adapter_names)):
            adapter_names[i] = "trained_" + adapter_names[i]
        # llama.model.load_adapter(lora_path, adapter_name="m_lora")
        # llama.model = PeftModel.from_pretrained(llama.model, trained_peft_path, adapter_name=adapter_name) #, use_safetensors=True)
        # # llama.model = llama.model.merge_and_unload()
        # llama.model.load_adapter(trained_peft_path, adapter_name=adapter_name)
        llama.model.set_adapter([adapter_names])
        llama.model = convert_model_adapter_params_to_torch_dtype(model=llama.model, peft_name="trained_", torch_dtype=torch_dtype)
        llama.model = llama.model.to(torch_dtype)
    # else:
    #     # Method 2 - AdapterHub
    #     adapters.init(model=llama.model)
    #     loaded_peft = llama.model.load_adapter(trained_peft_path, with_head=False)
    #     llama.model.set_active_adapters([loaded_peft])
    #     llama.model.adapter_to(loaded_peft, device=device)
    #
    #     llama.model = llama.model.to(torch.bfloat16)
    #
    #     llama.model.enable_input_require_grads()
    #     llama.model.gradient_checkpointing_enable()
    #     logger.info("\nMethod 2 LLaMA Model's Summary:\n{}\n\n\n".format(llama.model.adapter_summary()))

    logger.info("Loaded MODEL: \n{}".format(llama.model))

    if CHAT_TEMPLATE:
        logger.info("****** RESULTS ARE GENERATED USING CHAT TEMPLATE ******")

    data = SumDataLoader(dataset_name=domain, training_samples=training_samples, eval_samples=eval_samples,
                         test_samples=test_samples, sort_dataset_on_article_len=sort_data, chat_template=CHAT_TEMPLATE)
    data.loading_dataset_splits()

    data.train_set = None
    data.validation_set = None

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, test_samples=test_samples,
                  peft_full_name=all_adapters, device=device)
