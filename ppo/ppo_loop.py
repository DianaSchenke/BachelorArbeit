import torch
from trl.trainer import peft_module_casting_to_bf16
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from fixed_ppo_trainer import PPOTrainerFixed #my fixed version of the PPOTrainer from TRL 0.8.6 that plays nice with unsloth models
from tqdm import tqdm
from accelerate import PartialState
from utils.eval_utils import get_length
from utils.dataset_utils import DatasetWrapper
from datetime import datetime
from unsloth import FastLanguageModel

# THIS FILE WAS BUILD FOR trl==0.8.6 AND DOESN'T RUN ON SOME NEWER VERSIONS OF TRL
# THE REST OF THE REPO SHOULD RUN ON UP-TO-DATE PACKAGE VERSIONS


torch.autograd.set_detect_anomaly(True)

now = datetime.now()
device_string = PartialState().process_index

language = 'en-US' # en-US, de

batch_size = 128
mini_batch_size = 2
epochs = 1
output_dir = "../output/ppo/"+now.strftime("%Y-%m-%d_%H-%M-%S")
ds = DatasetWrapper("dataset_files/64k_qaw_ppo")
ds.shuffle(42)
ds_train = ds.get_train_split(samples=8000)


max_seq_length = 2048


model_path = "/data/home/scl33452/PycharmProjects/Training-Setup/output/sft/qaw-2024-10-28_16-14-51/training/checkpoint-12000"

model, tokenizer = FastLanguageModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16)

tokenizer.padding_side='left'
tokenizer.pad_token_id = tokenizer.eos_token_id

# wrapping a AutoModelForCausalLMWithValueHead around a FastLanguageModel isn't really supported,
# I do it anyway and fix the issues with a bunch of hacky solutions
model = AutoModelForCausalLMWithValueHead(model)
model.is_peft_model = True
peft_module_casting_to_bf16(model)

def get_max_len(dataset):
    max_len = -1
    for sample in tqdm(dataset):
        tokenized_query = tokenizer(sample["query"]["query"]).data["input_ids"]
        length = len(tokenized_query)
        if length > max_len:
            max_len = length
    return max_len

padding_width = get_max_len(ds_train)

config = PPOConfig(
    batch_size=batch_size,
    gradient_accumulation_steps=1,
    log_with='tensorboard',
    project_kwargs = {"logging_dir": output_dir+'/training'},
    tracker_project_name = "training",
    mini_batch_size= mini_batch_size,
    ppo_epochs= 4,
)

ppo_trainer = PPOTrainerFixed(config=config, model=model, tokenizer=tokenizer, dataset=ds_train)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": max_seq_length,
    "use_cache": True,
    "cache_implementation" : "dynamic"
}






for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        #this fixes an issue with the base_model that unsloth wraps around misbehaving
        #I assume that some incompatibility/bug in the PPO Trainer or AutoModelForCausalLMWithValueHead falsely sets this value to True when it definitely should be False
        ppo_trainer.model.pretrained_model.base_model.model.base_model.training = False

        #prepare inputs
        inputs = ppo_trainer.tokenizer(batch["query"]["query"], return_tensors="pt", padding="max_length",max_length=padding_width)
        input_list = [tensor for tensor in inputs.data["input_ids"]]

        #sample model
        with torch.no_grad():
            ppo_trainer.model.gradient_checkpointing_disable()
            response_tensor = ppo_trainer.generate(input_list, return_prompt=False, **generation_kwargs)

        #decode responses and get rewards
        rewards = []
        responses = []
        eval_modes = batch["query"]["eval_mode"]
        lengths = batch["query"]["length"]
        for i, response in enumerate(response_tensor):
            response_txt = tokenizer.decode(response[:-1]).lstrip()
            actual_length = get_length(response_txt, eval_mode=eval_modes[i], language=language)
            actual_length_tensor = torch.tensor(actual_length, dtype=torch.bfloat16, device=model.pretrained_model.device)
            length_diff = lengths[i].to(torch.bfloat16) - actual_length_tensor
            reward = torch.absolute(length_diff)
            rewards.append(reward)
            responses.append(response)

        #update model
        ppo_trainer.step(queries=input_list, responses=responses, scores=rewards)
ppo_trainer.save_pretrained(save_directory=output_dir+"/model")