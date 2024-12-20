from unsloth import FastLanguageModel
from orpo_loop import do_orpo
from datetime import datetime
from utils.dataset_utils import DatasetWrapper
import torch

now = datetime.now()
output_dir = "../output/orpo/qaw_3_epochs_sft"+now.strftime("%Y-%m-%d_%H-%M-%S")
max_seq_length = 2048


#model_path = "/data/home/scl33452/PycharmProjects/Training-Setup/output/sft/qaw-2024-10-28_16-14-51/training/checkpoint-12000"
model_path = "/data/home/scl33452/PycharmProjects/Training-Setup/output/sft/lr-2024-10-28_16-21-24/model"

model, tokenizer = FastLanguageModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16)
tokenizer.pad_token_id = tokenizer.eos_token_id

#ds = DatasetWrapper("dataset_files/32k_qaw_preference_3_epochs_sft")
ds = DatasetWrapper("dataset_files/32k_qaw_llama_preference")
ds.shuffle(42)
ds_train = ds.get_train_split(samples=32000)

do_orpo(model, tokenizer, output_dir, ds_train, epochs=1)