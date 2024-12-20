import pathlib
from transformers import AutoTokenizer, pipeline
import torch
from utils.eval_utils import split_messages, get_length
from unsloth import FastLanguageModel

llama_3_1_base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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")
base_dir = str(pathlib.Path(__file__).resolve().parent.parent)


def get_trained_model(path):
    model, tokenizer = FastLanguageModel.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)

question = "What is the air-speed velocity of an unladen swallow?"

model = get_trained_model(base_dir+"/output/orpo/qaw_3_epochs_sft2024-11-09_10-00-28/model")

for i in [10, 50, 150, 250]:
    input = question + f' Generate precisely {i} characters in your response.'
    input_str = llama_3_1_base_tokenizer.apply_chat_template([{"role": "user", "content": input}],tokenize=False, add_generation_prompt=True)
    output = split_messages(model(input_str)[0]["generated_text"])[-1]
    print("Q:  " + input)
    print("R:  " + output)
    print(f"actual response length = {get_length(output, 'chars', 'en-US')} chars\n")