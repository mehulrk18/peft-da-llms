import argparse
import logging

import adapters
import pandas as pd
import torch
from transformers import AutoTokenizer

from dataset_lib import inference_prompt, SumDataLoader
from utils import generate_summary, get_pretrained_model, MODEL_ID, rouge_metric, LLaMAModelClass

global CHAT_TEMPLATE


def testing_model(llama_model, llama_tokenizer, domain, training_samples, eval_samples, test_samples,
                  sort_data, peft_full_name, device):
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
        with open("random_ip_summaries/random_text_{}_pipeline.txt".format(peft_full_name), "w") as f:
            f.write("Wikipedia Article: \n{} \n\n\n\n Summary:{}\n".format(random_text, summ))
            logger.info("Written Random article summary")
    except Exception as e:
        logger.error("Exception: ".format(e))
        pass

    data = SumDataLoader(dataset_name=domain, training_samples=training_samples, eval_samples=eval_samples,
                         test_samples=test_samples, sort_dataset_on_article_len=sort_data, chat_template=CHAT_TEMPLATE)
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
    min_samples = min(test_samples, len(df_test_data))
    for i in range(min_samples):
        summary = generate_summary(model=llama_model, tokenizer=llama_tokenizer, content=df_test_data["article"][i],
                                   device=device)
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

    parser.add_argument("--checkpoint",type=str, default=None, help="Path of the PT Model Checkpoint to be loaded." )
    parser.add_argument("--trained_peft_path", type=str, help="Path of the PEFT to be loaded.")
    parser.add_argument("--training_samples", type=int, default=1, help="Number of training Samples")
    parser.add_argument("--eval_samples", type=int, default=1, help="Number of Evaluation Samples")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of Samples to be tested")
    parser.add_argument("--sorted_dataset", type=bool, default=False, help="do you want to sort the dataset?")
    parser.add_argument("--chat_template", type=bool, default=False, help="Using chat template for tokenizing")
    # parser.add_argument("--ah", type=bool, help="Load Model and Adapter from Adapter HUB")
    # parser.add_argument("--domain", type=str, help="Domain name for dataset")

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        main_directory = "/content/drive/My Drive/Colab Notebooks/"
    except Exception as e:
        main_directory = ""

    args = parser.parse_args()
    trained_peft_path = main_directory + args.trained_peft_path
    mlm = True if "mlm" in trained_peft_path else False
    model_checkpoint = args.checkpoint
    training_samples = args.training_samples
    eval_samples = args.eval_samples
    test_samples = args.test_samples
    sort_data = args.sorted_dataset
    CHAT_TEMPLATE = True if "chat_template" in trained_peft_path or args.chat_template else False
    use_instruct_model = True if "instruct" in trained_peft_path else False

    peft_path_splits = trained_peft_path.split("/")
    if peft_path_splits[0] == "results":
        peft_dir = "_".join(peft_path_splits)
        domain = peft_path_splits[3].split("_")[0]
        peft_name = peft_path_splits[3]

    elif trained_peft_path.split("/")[0] == "saved_models":
        peft_dir = peft_path_splits[1]
        provider, domain, peft_type = tuple(peft_dir.split("_")[:3])
        peft_name = "{}_{}".format(domain, peft_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(
        filename=main_directory + 'logs/testing_{}.log'.format("_".join(peft_path_splits)),  # The log file to write to
        filemode='w',  # Overwrite the log file each time the script runs
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s -\n%(message)s'  # Log message format
    )
    logger = logging.getLogger()

    # llama_model = get_pretrained_model(ah=ah)
    llama = LLaMAModelClass(version=3.0, instruct_mode=use_instruct_model, quantization_config=None,
                            model_checkpoint=model_checkpoint, mlm=mlm)
    # llama = LLaMAModelClass(version=3.0, instruct_mode=False, quantization_config=None)

    logger.info("Check point MODEL: \n{}".format(llama.model))
    # domain = args.domain

    # if ah:
    #     loaded_peft = llama_model.load_adapter(trained_peft_path, with_head=True)
    #     llama_model.set_active_adapters(loaded_peft)
    #     llama_model.adapter_to(loaded_peft, device=device)
    #
    #     llama_model = llama_model.to(torch.bfloat16)
    #
    #     llama_model.enable_input_require_grads()
    #     llama_model.gradient_checkpointing_enable()
    #
    #     logger.info("\nLLaMA Model's Summary:\n", llama_model.adapter_summary())
    #
    # else:
        # Method 1
        # from peft import PeftModel
        # llama_model = PeftModel.from_pretrained(llama_model, trained_peft_path, adapter_name=peft_name)
        # llama_model = llama_model.merge_and_unload()
        # llama_model.load_adapter(trained_peft_path, adapter_name=peft_name)
        # llama_model.set_adapter(peft_name)
    # for name, param in llama_model.named_parameters():
    #     if "lora" in name:
    #         logger.info(name, param.dtype)
    #         param.data = param.data.to(torch.bfloat16)
        # if param.ndim == 1:
        #     # cast the small parameters (e.g. layernorm) to fp32 for stability
        #     logger.info(name, param.dtype)
            # param.data = param.data.to(torch.float32)

        # Method 2
    adapters.init(model=llama.model)
    loaded_peft = llama.model.load_adapter(trained_peft_path, with_head=False)
    llama.model.set_active_adapters([loaded_peft])
    llama.model.adapter_to(loaded_peft, device=device)

    llama.model = llama.model.to(torch.bfloat16)

    llama.model.enable_input_require_grads()
    llama.model.gradient_checkpointing_enable()
    logger.info("\nMethod 2 LLaMA Model's Summary:\n{}\n\n\n".format(llama.model.adapter_summary()))

    logger.info("Loaded MODEL: \n{}".format(llama.model))

    if CHAT_TEMPLATE:
        logger.info("****** RESULTS ARE GENERATED USING CHAT TEMPLATE ******")

    # for name, param in llama.model.named_parameters():
    #     logger.info(f"Parameter: {name} | Device: {param.device}")
    #
    # llama_tokenizer = AutoTokenizer.from_pretrained(
    #     MODEL_ID,
    #     padding_side="right",
    #     tokenizer_type="llama",
    #     trust_remote_code=True,
    #     use_fast=True
    # )

    testing_model(llama_model=llama.model, llama_tokenizer=llama.tokenizer, domain=domain,
                  training_samples=training_samples, eval_samples=eval_samples, test_samples=test_samples,
                  sort_data=sort_data, peft_full_name=peft_dir, device=device)
