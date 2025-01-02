import evaluate


def rouge_metric():
    rouge = evaluate.load("rouge")
    return rouge


def bertscore_metric():
    bertscore = evaluate.load("bertscore")
    return bertscore


def bleu_metric():
    bleu = evaluate.load("bleu")
    return bleu


def bleurt_metric():
    bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="BLEURT-20")
    return bleurt


def dvo_metric():
    pass


def factscore_metric():
    pass
