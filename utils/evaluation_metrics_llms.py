import evaluate


def rouge_metric():
    rouge = evaluate.load("rouge")
    return rouge


def bertscore_metric():
    pass


def bleu_metric():
    pass


def dvo_metric():
    pass

