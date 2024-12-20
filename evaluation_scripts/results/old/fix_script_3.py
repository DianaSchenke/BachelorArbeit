import os
from utils.eval_utils import compare_responses_to_base_llama, create_base_llama_responses

dir_path= "/evaluation_scripts/results/old/qaw_sft_fixed_2"
save_path="/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_sft_fixed_3"
os.makedirs(save_path, exist_ok=True)

#create_base_llama_responses(dir_path=dir_path, batch_size=32)
compare_responses_to_base_llama(dir_path=dir_path, save_path=save_path)