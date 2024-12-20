from evaluator import ModelEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.dataset_utils import DatasetWrapper
from utils.eval_utils import compare_responses_to_base_llama, create_base_llama_responses, BestOfNModelWrapper
from utils.custom_llama_3_1 import CustomLlama31
import torch
import pathlib
import os
from unsloth import FastLanguageModel


base_dir = str(pathlib.Path(__file__).resolve().parent.parent)

data_path_qaw = base_dir+"/base_datasets/questions_about_the_world"


save_base_path = base_dir+"/evaluation_scripts/results/naive_methods"
os.makedirs(save_base_path, exist_ok=True)


data_wrapper_qaw = DatasetWrapper(data_path_qaw)

data_wrapper_qaw.shuffle(42)

batch_size = 32
samples = 1280
ds_qaw = data_wrapper_qaw.get_eval_split(samples=samples)



language = 'en-US'
eval_modes = ['chars', "letters", "spoken", "space"]

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")

def get_trained_model(path, best_of_n=False, customLlama=False, temp_min=0.5, temp_max=3, n=2):
    model, tokenizer = FastLanguageModel.from_pretrained(path, dtype=torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)
    if best_of_n is False and customLlama is False:
        return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    if best_of_n is True and customLlama is False:
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
        return BestOfNModelWrapper(pipe, temp_min=temp_min, temp_max=temp_max, n=n)
    if best_of_n is False and customLlama is True:
        return CustomLlama31(model=model, tokenizer=tokenizer)
    if best_of_n is True and customLlama is True:
        pipe = CustomLlama31(model=model, tokenizer=tokenizer)
        return BestOfNModelWrapper(pipe, temp_min=temp_min, temp_max=temp_max, n=n, use_custom_llama=True)

def get_base_model(best_of_n=False, customLlama=False, temp_min=0.5, temp_max=3, n=2):
    if best_of_n is False and customLlama is False:
        pipe = pipeline(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=2048,
                        model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        return pipe
    if best_of_n is True and customLlama is False:
        pipe = pipeline(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=2048,
                        model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        return BestOfNModelWrapper(pipe, temp_min=temp_min, temp_max=temp_max, n=n)
    if customLlama is True:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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", device_map="auto")
        pipe = CustomLlama31(model=model, tokenizer=llama_3_1_base_tokenizer)
        if best_of_n is True:
            return BestOfNModelWrapper(pipe, temp_min=temp_min, temp_max=temp_max, n=n, use_custom_llama=True)
        else:
            return pipe



#model_evaluator = ModelEvaluator(pipeline=get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model", best_of_n=True),
#                                 eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/best_of_2.df", ds=ds_qaw,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, best_of_n_mode=True)
#model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model", customLlama=True),
#                                 eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/eos_prob.df", ds=ds_qaw,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, custom_llama_mode=True)
#model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model", best_of_n=True, customLlama=True),
#                                 eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/best_of_2_and_eos_prob.df", ds=ds_qaw,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, best_of_n_mode=True, custom_llama_mode=True)
#model_evaluator.eval_model()


#model_evaluator = ModelEvaluator(pipeline=get_base_model(best_of_n=True),
#                                 eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/best_of_2_baseline.df", ds=ds_qaw,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, best_of_n_mode=True)
#model_evaluator.eval_model()

model_evaluator = ModelEvaluator(pipeline=get_base_model(customLlama=True),
                                 eval_modes=eval_modes, language=language, batch_size=batch_size,
                                 samples=samples, save_loc=save_base_path+f"/eos_prob_baseline.df", ds=ds_qaw,
                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, custom_llama_mode=True)
model_evaluator.eval_model()

model_evaluator = ModelEvaluator(pipeline=get_base_model(best_of_n=True, customLlama=True),
                                 eval_modes=eval_modes, language=language, batch_size=batch_size,
                                 samples=samples, save_loc=save_base_path+f"/best_of_2_and_eos_prob_baseline.df", ds=ds_qaw,
                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer, best_of_n_mode=True, custom_llama_mode=True)
model_evaluator.eval_model()