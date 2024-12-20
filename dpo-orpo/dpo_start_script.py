from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from dpo_loop import do_dpo
from datetime import datetime
from utils.dataset_utils import DatasetWrapper
import torch

now = datetime.now()
output_dir = "../output/dpo/qaw_3_epochs_sft"+now.strftime("%Y-%m-%d_%H-%M-%S")
max_seq_length = 2048


model_path = "/data/home/scl33452/PycharmProjects/Training-Setup/output/sft/qaw-2024-10-28_16-14-51/training/checkpoint-12000"

model, tokenizer = FastLanguageModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16)
tokenizer.pad_token_id = tokenizer.eos_token_id

ds = DatasetWrapper("dataset_files/32k_qaw_preference_3_epochs_sft")
ds.shuffle(42)
ds_train = ds.get_train_split(samples=32000)

do_dpo(model, tokenizer, output_dir, ds_train, epochs=1)