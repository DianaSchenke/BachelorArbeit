from datasets import load_dataset
from utils.dataset_utils import create_base_dataset
ds = load_dataset("liselot321/newspaper_summaries_evaluations")

target_len = 30

def apply_template(data):
    summary = data["summary_original"]
    article = data["article"]
    input = "Summarize the following article:\n\n"+article+f"\n\nYour summary must be at most {target_len} cms of 12pt Times New Roman long."
    return {"input" : [input], "target_output" : summary}

ds = ds.map(apply_template)
ds = ds.remove_columns(["summary_llama", "summary_bart", "summary_original", "eval_sum_llama", "eval_sum_bart", "eval_sum_original"])
create_base_dataset(ds["test"], "newspaper_summaries")