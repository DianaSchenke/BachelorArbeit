from evaluate import load
bertscore = load("bertscore")


def get_bert_score(predictions, references, language="en"):
    return bertscore.compute(predictions=predictions, references=references, lang=language)
