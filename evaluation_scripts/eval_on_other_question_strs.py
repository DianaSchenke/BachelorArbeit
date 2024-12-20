from evaluator import ModelEvaluator
from transformers import AutoTokenizer, pipeline
from utils.dataset_utils import DatasetWrapper
import torch
import pathlib
import os
from unsloth import FastLanguageModel


base_dir = str(pathlib.Path(__file__).resolve().parent.parent)

data_path_qaw = base_dir+"/base_datasets/questions_about_the_world"


save_base_path = base_dir+"/evaluation_scripts/results/other_question_strs"
os.makedirs(save_base_path, exist_ok=True)


data_wrapper_qaw = DatasetWrapper(data_path_qaw)

data_wrapper_qaw.shuffle(42)

batch_size = 32
samples = 1280
ds_qaw = data_wrapper_qaw.get_eval_split(samples=samples)



language = 'en-US'
eval_modes = ['chars', "letters", "spoken", "space"]

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")

def get_trained_model(path):
    model, tokenizer = FastLanguageModel.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)

def get_base_model():
    pipe = pipeline(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=2048,
                        model_kwargs={"torch_dtype": torch.bfloat16}, device_map = "auto")
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe

model_evaluator = ModelEvaluator(pipeline=get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model"), eval_modes=eval_modes, language=language, batch_size=batch_size,
                                 samples=samples, save_loc=save_base_path+f"/trained.df", ds=ds_qaw,
                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, use_eval_question_strs=True)
model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_base_model(), eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/base.df", ds=ds_qaw,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, use_eval_question_strs=True)
#model_evaluator.eval_model()