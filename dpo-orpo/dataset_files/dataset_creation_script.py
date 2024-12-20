from utils.dataset_utils import DatasetMakerDPO
from transformers import AutoTokenizer, pipeline
import torch
from unsloth import FastLanguageModel

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")

#model_path = "/data/home/scl33452/PycharmProjects/Training-Setup/output/sft/qaw-2024-10-28_16-14-51/training/checkpoint-12000"
model_path = "/data/home/scl33452/PycharmProjects/Training-Setup/output/sft/lr-2024-11-08_14-17-30/model"
model, tokenizer = FastLanguageModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16)
tokenizer.pad_token_id = tokenizer.eos_token_id
FastLanguageModel.for_inference(model)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)


batch_size = 32
language = 'en-US' # en-US, de
eval_modes = ['chars', "letters", "spoken", "space"]

dataset_maker = DatasetMakerDPO(eval_modes=eval_modes, pipe=pipe, language=language, tokenizer=llama_3_1_base_tokenizer)
dataset_maker.make_data(train_start_sample=128000, train_samples=32000, eval_start_sample=1024, eval_samples=1024, test_samples=0)
dataset_maker.save_data("32k_qaw_llama_preference")