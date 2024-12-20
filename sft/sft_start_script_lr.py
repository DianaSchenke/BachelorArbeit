from sft_loop import do_sft

from utils.dataset_utils import DatasetWrapper
from datetime import datetime
now = datetime.now()

max_seq_length = 2048
#output_dir = "../output/sft/"+now.strftime("lr-%Y-%m-%d_%H-%M-%S")
output_dir = "../output/sft/lr-2024-11-08_14-17-30"
epochs = 3

ds = DatasetWrapper("dataset_files/128k_llama_response_sft")
ds.shuffle(42)
ds_train = ds.get_train_split(samples=128000)
ds_eval = ds.get_eval_split(samples=1024)


do_sft(ds_train, ds_eval, max_seq_length, output_dir, epochs, resume_from_checkpoint=True)