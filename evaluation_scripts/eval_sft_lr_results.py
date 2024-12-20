from evaluator import ModelEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.dataset_utils import DatasetWrapper
import torch
import pathlib
import os
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel

base_dir = str(pathlib.Path(__file__).resolve().parent.parent)
data_path = base_dir+"/base_datasets/questions_about_the_world"
model_base_path = base_dir+"/output/sft/lr-2024-10-28_16-21-24/training/"
save_base_path = base_dir+"/evaluation_scripts/results/lr_sft"
os.makedirs(save_base_path, exist_ok=True)


data_wrapper = DatasetWrapper(data_path)
data_wrapper.shuffle(42)

batch_size = 32
samples = 1280
ds = data_wrapper.get_eval_split(samples=samples)
language = 'en-US'
eval_modes = ['chars', "letters", "spoken", "space"]

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")


model_name_list = [
    "checkpoint-4000",
    "checkpoint-8000",
    "checkpoint-12000",
    "checkpoint-16000",
    "checkpoint-20000",
    "checkpoint-24000",
    "checkpoint-28000",
    "checkpoint-32000",
    "checkpoint-36000",
    "checkpoint-40000",
]
def get_trained_model(n):
    path = model_base_path+model_name_list[n]
    model, tokenizer = FastLanguageModel.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    return pipe



result_list = []
for i in range(10):
    model_evaluator = ModelEvaluator(pipeline=get_trained_model(i), eval_modes=eval_modes, language=language, batch_size=batch_size, samples=samples, save_loc=save_base_path+f"/{i+1}_epochs_sft.df", ds=ds, tokenizer_for_chat_template=llama_3_1_base_tokenizer)
    result_list.append(model_evaluator.eval_model())


