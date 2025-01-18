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
    bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="bleurt-large-128")
    return bleurt


def meteor_metric():
    meteor = evaluate.load("meteor")
    return meteor


def factscore_metric():
    print("To run factscore use factscore_main.py")
    pass
