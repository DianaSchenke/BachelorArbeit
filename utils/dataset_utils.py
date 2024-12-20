from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

from utils.eval_utils import get_length, format_question, split_messages
import random
from abc import ABC, abstractmethod
import pathlib

dir_path = str(pathlib.Path(__file__).parent.parent.resolve())



def create_base_dataset(ds, save_loc):
    '''
    simple function that splits a huggingface dataset into train, eval and test splits, which are stored in separate
    files to avoid mixups.
    Test splits aren't actually used much in this project, as the eval splits proved to be big enough to just be sampled
    at different indexes when new data was needed.
    '''

    #THESE VALUES ARE HARDCODED HERE FOR CONSISTENCY
    train_ratio = .8
    eval_ratio = .1
    test_ratio = .1

    train_split_size = int(train_ratio * len(ds))
    eval_split_size = int(eval_ratio * len(ds))
    test_split_size = int(test_ratio * len(ds))
    train_split = Dataset.from_dict(ds[:train_split_size])
    eval_split = Dataset.from_dict(ds[train_split_size:train_split_size + eval_split_size])
    test_split = Dataset.from_dict(ds[train_split_size + eval_split_size:train_split_size + eval_split_size + test_split_size])
    train_split.save_to_disk(save_loc + "/train.hf")
    eval_split.save_to_disk(save_loc + "/eval.hf")
    test_split.save_to_disk(save_loc + "/test.hf")

def clean_and_reformat_ultrachat(data):
    # this function removes samples from ultrachat that have very short questions, which can cause issues elsewhere.
    cleaned_data = []
    first_turn = []
    question = []
    for sample in data:
        if len(sample[0]) > 30:
            cleaned_data.append(sample)
            question.append(sample[0])
            first_turn.append(sample[0] + " " + sample[1])

    return Dataset.from_dict({"data": cleaned_data, "question": question, "first_turn": first_turn})

class DatasetWrapper:
    '''
    custom class to load and shuffle datasets stored in the base_datasets folder for further use
    includes some utilities like loading parts of datasets based on indexes
    '''
    def __init__(self, loc):
        try:
            self.train_split = load_from_disk(loc + "/train.hf")
        except FileNotFoundError:
            self.train_split = None
        try:
            self.eval_split = load_from_disk(loc + "/eval.hf")
        except FileNotFoundError:
            self.eval_split = None
        try:
            self.test_split = load_from_disk(loc + "/test.hf")
        except FileNotFoundError:
            self.test_split = None

    def shuffle(self, seed):
        if self.train_split is not None:
            self.train_split = self.train_split.shuffle(seed)
        if self.eval_split is not None:
            self.eval_split = self.eval_split.shuffle(seed)
        if self.test_split is not None:
            self.test_split = self.test_split.shuffle(seed)

    def get_train_split(self, samples=None, start_sample=0):
        if samples is not None and self.train_split is not None:
            assert len(self.train_split) >= start_sample+samples
            return Dataset.from_dict(self.train_split[start_sample:start_sample+samples])
        else:
            return self.train_split

    def get_eval_split(self, samples=None, start_sample=0):
        if samples is not None and self.eval_split is not None:
            assert len(self.eval_split) >= start_sample+samples
            return Dataset.from_dict(self.eval_split[start_sample:start_sample+samples])
        else:
            return self.eval_split

    def get_test_split(self, samples=None, start_sample=0):
        if samples is not None and self.test_split is not None:
            assert len(self.test_split) >= start_sample+samples
            return Dataset.from_dict(self.test_split[start_sample:start_sample+samples])
        else:
            return self.test_split


class DatasetMaker(ABC):
    '''
    Abstract class for making datasets tailored for fine-tuning Trainer classes. Actual Implementations are further
    below and are tailored specifically to work with UltraChat.
    '''
    def __init__(self, seed=42, dataset = "questions_about_the_world" ):
        self.ds = DatasetWrapper(dir_path + "/base_datasets/" + dataset)
        self.seed = seed

    def make_data(self, train_samples=None, train_start_sample=0, eval_samples=None, eval_start_sample=0, test_samples=None, test_start_sample=0, shuffle_first=False, seed=42):
        if shuffle_first:
            self.ds.shuffle(seed)

        if train_samples != 0 and self.ds.train_split is not None:
            self.ds.train_split = self.data_procedure(self.ds.get_train_split(train_samples, train_start_sample))
        else:
            self.ds.train_split = None
        if eval_samples != 0 and self.ds.eval_split is not None:
            self.ds.eval_split = self.data_procedure(self.ds.get_eval_split(eval_samples, eval_start_sample))
        else:
            self.ds.eval_split = None
        if test_samples != 0 and self.ds.test_split is not None:
            self.ds.test_split = self.data_procedure(self.ds.get_test_split(test_samples, test_start_sample))
        else:
            self.ds.test_split = None

    def save_data(self, loc):
        if self.ds.train_split is not None:
            self.ds.train_split.save_to_disk(loc + "/train.hf")
        if self.ds.eval_split is not None:
            self.ds.eval_split.save_to_disk(loc + "/eval.hf")
        if self.ds.test_split is not None:
            self.ds.test_split.save_to_disk(loc + "/test.hf")

    @abstractmethod
    def data_procedure(self, dataset):
        pass




class DatasetMakerSFT(DatasetMaker):
    # Class that reformats a data from UltraChat to work with the SFT Trainer from transformers.
    def __init__(self, eval_modes, language, tokenizer, seed=42, dataset = "questions_about_the_world"):
        super().__init__(seed, dataset)
        self.eval_modes = eval_modes
        self.language = language
        self.tokenizer = tokenizer

    def apply_template(self, data):
        eval_mode = random.choice(self.eval_modes)
        question = data["question"]
        answer_str = data["first_turn"].replace(question, "").lstrip()
        length = get_length(answer_str, eval_mode, self.language)
        question_str = format_question(question, eval_mode, length)
        messages = [{"role": "user", "content": question_str},{"role": "assistant", "content": answer_str}]
        messages_with_special_tokens = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"messages" : messages_with_special_tokens}

    def data_procedure(self, dataset):
        if "id" in dataset.features:
            dataset = dataset.remove_columns(["id"])
        if "data" in dataset.features:
            dataset = dataset.remove_columns(["data"])
        dataset = dataset.map(self.apply_template)
        dataset = dataset.remove_columns(["question", "first_turn"])
        return dataset


class DatasetMakerPPO(DatasetMaker):
    # Class that reformats a data from UltraChat to work with the PPO Trainer from TRL 8.6.0.
    def __init__(self, eval_modes, language, tokenizer, seed=42, dataset = "questions_about_the_world"):
        super().__init__(seed, dataset)
        self.eval_modes = eval_modes
        self.language = language
        self.tokenizer = tokenizer

    def apply_template(self, data):
        eval_mode = random.choice(self.eval_modes)
        question = data["question"]
        answer_str = data["first_turn"].replace(question, "").lstrip()
        length = get_length(answer_str, eval_mode, self.language)
        question_str = format_question(question, eval_mode, length)
        query = self.tokenizer.apply_chat_template([{"role": "user", "content": question_str}], tokenize=False, add_generation_prompt=True)
        return {'query': {'query' : query, 'length' : length, 'eval_mode' : eval_mode}}

    def data_procedure(self, dataset):
        if "id" in dataset.features:
            dataset = dataset.remove_columns(["id"])
        if "data" in dataset.features:
            dataset = dataset.remove_columns(["data"])
        dataset = dataset.map(self.apply_template)
        dataset = dataset.remove_columns(["question", "first_turn"])
        return dataset


class DatasetMakerDPO(DatasetMaker):
    '''
    Class that creates a preference dataset based on the question in UltraChat to work with the DPO and ORPO Trainer
    from TRL. The model pipeline passed using the "pipe" argument should contain the same model that is to be trained
    with DPO/ORPO.
    '''
    def __init__(self, eval_modes, language, pipe, tokenizer, seed=42, batch_size = 32, dataset = "questions_about_the_world"):
        super().__init__(seed, dataset)
        self.eval_modes = eval_modes
        self.language = language
        self.pipe = pipe
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def apply_template(self, data):
        eval_mode = random.choice(self.eval_modes)
        question = data["question"]
        answer_str = data["first_turn"].replace(question + "\n", "")
        length = get_length(answer_str, eval_mode, self.language)
        question_str = format_question(question, eval_mode, length)
        input = [
            {"role": "user", "content": question_str},
        ]
        input_with_special_tokens = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        return {"input": input_with_special_tokens, "length": length, "eval_mode" : eval_mode}

    def get_output(self, batch, temp):
        preds = []
        for i, output in enumerate(
                self.pipe(KeyDataset(batch, "input"), batch_size=self.batch_size, truncation="only_first", temperature=temp)):
            output_str = output[0]["generated_text"]
            message_list = split_messages(output_str)
            pred = message_list[-1]
            preds.append(pred)
        return preds

    def data_procedure(self, dataset):
        #prepare dataset
        if "id" in dataset.features:
            dataset = dataset.remove_columns(["id"])
        if "data" in dataset.features:
            dataset = dataset.remove_columns(["data"])
        dataset = dataset.map(self.apply_template)
        dataset = dataset.remove_columns(["first_turn"])


        data = {"prompt": [], "chosen": [], "rejected": []}
        for idx in tqdm(range(0, len(dataset), self.batch_size)):
            #get two different predictions
            batch = Dataset.from_dict(dataset[idx: idx + self.batch_size])
            preds_1 = self.get_output(batch, 1.25)
            preds_2 = self.get_output(batch, 0.75)
            for i in range(len(batch)):
                #pick answer that is closer to target to be preferred
                diff_1 = abs(batch["length"][i] - get_length(preds_1[i], batch["eval_mode"][i], self.language))
                diff_2 = abs(batch["length"][i] - get_length(preds_2[i], batch["eval_mode"][i], self.language))
                if diff_1 < diff_2:
                    chosen_answer = preds_1[i]
                    rejected_answer = preds_2[i]
                else:
                    chosen_answer = preds_2[i]
                    rejected_answer = preds_1[i]

                #adjust length requirement in prompt to match length of preferred response
                eval_mode = batch["eval_mode"][i]
                new_target_len = get_length(chosen_answer, eval_mode, self.language)
                new_question_str = format_question(batch["question"][i], eval_mode, new_target_len)
                prompt = [
                    {"role": "user", "content": new_question_str},
                ]
                prompt_with_special_tokens = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

                data["prompt"].append(prompt_with_special_tokens)
                data["chosen"].append(chosen_answer)
                data["rejected"].append(rejected_answer)
        return Dataset.from_dict(data)

