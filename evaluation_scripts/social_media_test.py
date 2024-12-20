from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import pathlib
from unsloth import FastLanguageModel
from utils.eval_utils import split_messages
from utils.dataset_utils import DatasetWrapper
import os
from datasets import Dataset

#DEPRECATED

base_dir = str(pathlib.Path(__file__).resolve().parent.parent)

data_path_qaw = base_dir+"/base_datasets/questions_about_the_world"


save_base_path = base_dir+"/evaluation_scripts/results/social_media_test"
os.makedirs(save_base_path, exist_ok=True)

data_wrapper_qaw = DatasetWrapper(data_path_qaw)

data_wrapper_qaw.shuffle(42)

batch_size = 2
samples = 1280
ds_qaw = data_wrapper_qaw.get_eval_split(samples=samples)

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")


model_assistant, tokenizer_assistant = FastLanguageModel.from_pretrained(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model", local_files_only=True, dtype=torch.bfloat16)
FastLanguageModel.for_inference(model_assistant)
assistant = pipeline(task="text-generation", model=model_assistant, tokenizer=tokenizer_assistant, max_new_tokens=2048)

model_user, tokenizer_user = FastLanguageModel.from_pretrained(model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", dtype=torch.bfloat16,)
FastLanguageModel.for_inference(model_user)
user = pipeline(task="text-generation", model=model_user, tokenizer=tokenizer_user, max_new_tokens=2048)

sys_prompt_assistant = "You are a helpful chatbot. All your responses should consist of 280 or less characters."

def generate_chat(start_prompt, iterations, sys_prompt_assistant):
    chat = [{"role": "user", "content": start_prompt}]
    for i in range(iterations):
        input_assistant = llama_3_1_base_tokenizer.apply_chat_template([{"role" : "system", "content" : sys_prompt_assistant }]+chat, tokenize=False, add_generation_prompt=True)
        output_assistant = assistant(input_assistant)[0]["generated_text"]
        messages = split_messages(output_assistant)
        response_assistant = messages[-1]
        chat.append({"role" : "assistant", "content" : response_assistant})

        input_user = llama_3_1_base_tokenizer.apply_chat_template(chat, tokenize=False)+"<|start_header_id|>user<|end_header_id|>\n\n"
        output_user = user(input_user)[0]["generated_text"]
        messages = split_messages(output_user)
        response_user = messages[-1]
        chat.append({"role" : "user", "content" : response_user})
    return chat





def generate_chat_batched(batch, iterations, sys_prompt_assistant):
    chats = []
    for start_prompt in batch["question"]:
        chats.append([{"role": "user", "content": start_prompt}])
    for i in range(iterations):
        assistant_inputs = []
        for chat in chats:
            assistant_inputs.append(llama_3_1_base_tokenizer.apply_chat_template([{"role": "system", "content": sys_prompt_assistant}] + chat, tokenize=False, add_generation_prompt=True))
        for i, output in enumerate(assistant(assistant_inputs, batch_size=batch_size, truncation="only_first")):
            messages = split_messages(output[0]["generated_text"])
            chats[i].append({"role" : "assistant", "content" : messages[-1]})

        user_inputs = []
        for chat in chats:
            user_inputs.append(llama_3_1_base_tokenizer.apply_chat_template(chat, tokenize=False)+"<|start_header_id|>user<|end_header_id|>\n\n")
        for i, output in enumerate(user(user_inputs, batch_size=batch_size, truncation="only_first")):
            messages = split_messages(output[0]["generated_text"])
            chats[i].append({"role" : "user", "content" : messages[-1]})
    return chats

chats = []
for idx in range(0,samples, batch_size):
    batch = ds_qaw[idx: idx + batch_size]
    chats += generate_chat_batched(batch, 10, sys_prompt_assistant)
    pass