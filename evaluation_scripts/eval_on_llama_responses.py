from evaluator import ModelEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.dataset_utils import DatasetWrapper
from utils.eval_utils import compare_responses_to_base_llama, create_base_llama_responses
import torch
import pathlib
import os
from unsloth import FastLanguageModel

base_dir = str(pathlib.Path(__file__).resolve().parent.parent)
data_path = base_dir+"/base_datasets/llama_response_data_full"
save_base_path = base_dir+"/evaluation_scripts/results/qaw_llama_responses_comparison"
os.makedirs(save_base_path, exist_ok=True)


data_wrapper = DatasetWrapper(data_path)
data_wrapper.shuffle(42)

batch_size = 32
samples = 1280
ds = data_wrapper.get_eval_split(samples=samples)
language = 'en-US'
eval_modes = ['chars', "letters", "spoken", "space"]

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")






def get_trained_model(path):
    model, tokenizer = FastLanguageModel.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    return pipe

def get_base_model():
    pipe = pipeline(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=2048,
                        model_kwargs={"torch_dtype": torch.bfloat16}, device_map = "auto")
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe


#model_evaluator = ModelEvaluator(pipeline=get_trained_model(base_dir+"/output/orpo/qaw_llama_3_epochs_sft2024-11-16_09-25-54/model"), eval_modes=eval_modes, language=language, batch_size=batch_size, samples=samples, save_loc=save_base_path+f"/trained_on_llama.df", ds=ds, tokenizer_for_chat_template=llama_3_1_base_tokenizer)
#model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model"), eval_modes=eval_modes, language=language, batch_size=batch_size, samples=samples, save_loc=save_base_path+f"/trained_on_gpt.df", ds=ds, tokenizer_for_chat_template=llama_3_1_base_tokenizer)
#model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_base_model(), eval_modes=eval_modes, language=language, batch_size=batch_size, samples=samples, save_loc=save_base_path+f"/baseline.df", ds=ds, tokenizer_for_chat_template=llama_3_1_base_tokenizer)
#model_evaluator.eval_model()

#create_base_llama_responses(dir_path=save_base_path, batch_size=32)
compare_responses_to_base_llama(dir_path=save_base_path, save_path=save_base_path)