from utils.dataset_utils import DatasetMakerSFT
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")

language = 'en-US' # en-US, de
eval_modes = ['chars', "letters", "spoken", "space"]
dataset_maker = DatasetMakerSFT( eval_modes=eval_modes, tokenizer=tokenizer, language=language, dataset = "llama_response_data_full")
dataset_maker.make_data(train_samples=128000, eval_samples=1024, test_samples=0)
dataset_maker.save_data("128k_llama_response_sft")