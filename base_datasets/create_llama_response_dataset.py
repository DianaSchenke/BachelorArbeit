from datasets import Dataset
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import random
from utils.dataset_utils import DatasetMaker
from transformers import pipeline
import torch

class DatasetMakerLlamaBased(DatasetMaker):
    '''
    This class is used to make a modified version of UltraChat containing the same questions but responses form base
    Llama 3.1. The resulting dataset is formatted in the same way as UltraChat so it can be used as a drop in
    replacement.
    '''
    def __init__(self, eval_modes, language, pipe, seed=42, batch_size = 32):
        super().__init__(seed)
        self.eval_modes = eval_modes
        self.language = language
        self.pipe = pipe
        self.batch_size = batch_size

    def apply_template(self, data):
        question = data["question"]

        # these are vague to increase reandomness of length requirements
        # specifying anything longer then "medium length" results in excessively long answers
        question_str = question + random.choice([
            ' Please write a very short answer.',
            ' Please write a short answer.',
            ' Please write a medium length answer.',
        ])
        input_str = [
            {"role": "user", "content": question_str},
        ]
        return {"input": input_str}

    def get_output(self, batch, temp):
        output_strs = []
        preds = []
        for i, output in enumerate(
                self.pipe(KeyDataset(batch, "input"), batch_size=self.batch_size, truncation="only_first", temperature=temp)):
            output_str = output[0]["generated_text"]
            pred = output_str[-1]["content"]
            output_strs.append(output_str)
            preds.append(pred)
        return preds

    def data_procedure(self, dataset):
        dataset = dataset.remove_columns(["id", "data"])
        dataset = dataset.map(self.apply_template)

        data = {"question": [], "first_turn": []}
        for idx in tqdm(range(0, len(dataset), self.batch_size)):
            temp = random.random()*2+1 #(between 1 and 3)
            batch = Dataset.from_dict(dataset[idx: idx + self.batch_size])
            preds = self.get_output(batch, temp)
            for i in range(len(batch)):
                data["question"].append(batch["question"][i])
                data["first_turn"].append(batch["question"][i]+" "+preds[i])
        return Dataset.from_dict(data)

pipe = pipeline(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=1028, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id



batch_size = 32
language = 'en-US' # en-US, de
eval_modes = ['chars']
dataset_maker = DatasetMakerLlamaBased(eval_modes=eval_modes, pipe=pipe, language=language)

dataset_maker.make_data(train_samples=32000, eval_samples=3200, test_samples=3200, shuffle_first=True, seed=42)
dataset_maker.save_data("llama_response_data_32k")