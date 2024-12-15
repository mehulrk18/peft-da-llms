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


def dvo_metric():
    pass

