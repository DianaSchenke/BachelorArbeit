from evaluator import ModelEvaluator
from transformers import AutoTokenizer, pipeline
from utils.dataset_utils import DatasetWrapper
import torch
import pathlib
import os
from unsloth import FastLanguageModel


base_dir = str(pathlib.Path(__file__).resolve().parent.parent)

data_path_aem = base_dir+"/base_datasets/assistance_on_existent_materials"
data_path_wac = base_dir+"/base_datasets/writing_and_creation"

save_base_path = base_dir+"/evaluation_scripts/results"
os.makedirs(save_base_path+"/aem_test", exist_ok=True)
os.makedirs(save_base_path+"/wac_test", exist_ok=True)


data_wrapper_aem = DatasetWrapper(data_path_aem)
data_wrapper_wac = DatasetWrapper(data_path_wac)
data_wrapper_aem.shuffle(42)
data_wrapper_wac.shuffle(42)

batch_size = 8
samples = 1280
ds_aem = data_wrapper_aem.get_eval_split(samples=samples)
ds_wac = data_wrapper_wac.get_eval_split(samples=samples)


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





model_evaluator = ModelEvaluator(pipeline= get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model"),
                                 eval_modes=eval_modes, language=language, batch_size=batch_size, samples=samples,
                                 save_loc=save_base_path+f"/aem_test/trained.df", ds=ds_aem,
                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer)
model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline= get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model"),
#                                 eval_modes=eval_modes, language=language, batch_size=batch_size, samples=samples,
#                                 save_loc=save_base_path+f"/wac_test/trained.df", ds=ds_wac,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer)
#model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_base_model(), eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/aem_test/base.df", ds=ds_aem,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer)
#model_evaluator.eval_model()

#model_evaluator = ModelEvaluator(pipeline=get_base_model(), eval_modes=eval_modes, language=language, batch_size=batch_size,
#                                 samples=samples, save_loc=save_base_path+f"/wac_test/base.df", ds=ds_wac,
#                                 tokenizer_for_chat_template=llama_3_1_base_tokenizer)
#model_evaluator.eval_model()