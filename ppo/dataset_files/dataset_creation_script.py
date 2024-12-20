from utils.dataset_utils import DatasetMakerPPO
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")
tokenizer.pad_token = tokenizer.eos_token

language = 'en-US' # en-US, de
eval_modes = ['chars', "letters", "spoken", "space"]

dataset_maker = DatasetMakerPPO(eval_modes=eval_modes, tokenizer=tokenizer, language=language)
dataset_maker.make_data(train_start_sample=128000, train_samples=64000, eval_start_sample=1024, eval_samples=1024, test_samples=0)
dataset_maker.save_data("64k_qaw_ppo")