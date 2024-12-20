from transformers import AutoTokenizer
from utils.dataset_utils import DatasetWrapper
import pathlib
import os
from utils.eval_utils import load_results
import pickle

base_dir = str(pathlib.Path(__file__).resolve().parent.parent)
data_path = base_dir+"/base_datasets/questions_about_the_world"
save_base_path = base_dir+"/evaluation_scripts/results/lr_sft_fixed"
os.makedirs(save_base_path, exist_ok=True)
data_wrapper = DatasetWrapper(data_path)
data_wrapper.shuffle(42)
llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")
samples = 1280
ds = data_wrapper.get_eval_split(samples=samples)

question_list = []
for sample in ds:
    question_list.append(llama_3_1_base_tokenizer.apply_chat_template([{"role": "user", "content": sample["question"]},], tokenize=False, add_generation_prompt=True))

dir_path = "/evaluation_scripts/results/old/lr_sft"
data_list = load_results(dir_path)
for df in data_list:
    df[1].insert(0, "base_question", question_list)
    for i in range(samples):
        question = df[1]["base_question"][i]
        messages = df[1]["messages"][i]
        test_question = question.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "")
        assert(test_question in messages)
    with open(save_base_path+"/"+df[0], mode="wb") as f:
        pickle.dump(df[1], file=f)