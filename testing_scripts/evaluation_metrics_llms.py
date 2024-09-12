import evaluate


def rouge_metric():
    rouge = evaluate.load("rouge")
    return rouge

