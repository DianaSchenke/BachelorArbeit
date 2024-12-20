from utils.sayit import sayit
from PIL import ImageFont
import random
from os import listdir
from os.path import isfile, join
import pickle
import re
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import numpy as np
from utils.bertscore import get_bert_score

font = ImageFont.truetype('times.ttf', 12)

def get_length(words, eval_mode, language):
    if eval_mode == "chars":
        return len(words)
    elif eval_mode == "letters":
        letters = [l for l in words if l.isalpha()]
        return len(letters)
    elif eval_mode == "spoken":
        return sayit(language, words)
    elif eval_mode == "space":
        len_px = font.getlength(words)
        len_pt = len_px * 0.75
        len_cm = len_pt / 28.35
        return  len_cm
    elif eval_mode == "words":
        word_list = words.strip().split(" ")
        word_count = len(word_list)
        return word_count

def format_question(question, eval_mode, length, eval_questions=False):
    #returns question string with a sentence attached at the end specifying the length requirement
    if eval_mode == "letters":
        if eval_questions:
            strs = [
            ' Your answer should have exactly {} letters.'.format(length),
            ' Please reply in precisely {} letters.'.format(length),
            ' Your response should have a length of {} letters.'.format(length),
            ' Respond within exactly {} letters.'.format(length),
            ' I need you to generate exactly {} letters in your reply.'.format(length),
            ]
        else:
            strs = [
            ' Please write an answer that consists of exactly {} letters.'.format(length),
            ' I need your reply to be {} letters long.'.format(length),
            ' Your response should have a length of {} letters.'.format(length),
            ' Please answer in exactly {} letters.'.format(length),
            ' Generate precisely {} letters in your response.'.format(length),
            ]
        return question + random.choice(strs)

    if eval_mode == "chars":
        if eval_questions:
            strs = [
            ' Your answer should have exactly {} characters.'.format(length),
            ' Please reply in precisely {} characters.'.format(length),
            ' Your response should have a length of {} characters.'.format(length),
            ' Respond within exactly {} characters.'.format(length),
            ' I need you to generate exactly {} characters in your reply.'.format(length),
            ]
        else:
            strs = [
            ' Please write an answer that consists of exactly {} characters.'.format(length),
            ' I need your reply to be {} characters long.'.format(length),
            ' Your response should have a length of {} characters.'.format(length),
            ' Please answer in exactly {} characters.'.format(length),
            ' Generate precisely {} characters in your response.'.format(length),
            ]
        return question + random.choice(strs)

    if eval_mode == "words":
        if eval_questions:
            strs = [
            ' Your answer should have exactly {} words.'.format(length),
            ' Please reply in precisely {} words.'.format(length),
            ' Your response should have a length of {} words.'.format(length),
            ' Respond within exactly {} words.'.format(length),
            ' I need you to generate exactly {} words in your reply.'.format(length),
            ]
        else:
            strs = [
            ' Please write an answer that consists of exactly {} words.'.format(length),
            ' I need your reply to be {} words long.'.format(length),
            ' Your response should have a length of {} words.'.format(length),
            ' Please answer in exactly {} words.'.format(length),
            ' Generate precisely {} characters in your words.'.format(length),
            ]
        return question + random.choice(strs)

    if eval_mode == "spoken":
        if eval_questions:
            strs = [
            ' Your answer should be able to be read out loud in {:.2f} seconds.'.format(length),
            ' Your reply should fit into a {:.2f} segment of a youtube video if read aloud.'.format(length),
            ' A talk show Host will read out your answer during her show so it needs to fill a {:.2f} seconds time slot.'.format(length),
            ' Produce an answer that fits into exactly {:.2f} seconds if I use it for a speech.'.format(length),
            ' Your response will be used to fill a TV segment of {:.2f} seconds where it will be read aloud.'.format(length),
            ]
        else:
            strs = [
            ' Please write an answer that I can read out loud in exactly {:.2f} seconds.'.format(length),
            ' Your response should take {:.2f} seconds if read aloud by a radio host.'.format(length),
            ' A TV presenter needs to be able to read out your answer within exactly {:.2f} seconds.'.format(length),
            ' Generate an answer that I can say in precisely {:.2f} seconds during my presentation.'.format(length),
            ' Your generated text will be read aloud during a speech and should therefore exactly fill the allocated time spot of {:.2f} seconds.'.format(length),
            ]
        return question + random.choice(strs)

    if eval_mode == "space":
        if eval_questions:
            strs = [
            ' I need your reply to be exactly {:.2f} cms of printed 12pt Times New Roman long.'.format(length),
            ' Your reply needs to fit into {:.2f} cms of 12pt Times New Roman long or it will be to long for the handout for my presentation.'.format(length),
            ' Generate a reply that is {:.2f} cms long if printed out and 12pt Times New Roman is used as the font.'.format(length),
            ' Your response will be printed in 12pt Times New Roman and should be {:.2f} cms long.'.format(length),
            ' Your reply will be printed in a newspaper using 12pt Times New Roman and fit within {:.2f} cms of space.'.format(length),
            ]
        else:
            strs = [
            ' Please write an answer that can fit exactly within {:.2f} cms of printed 12pt Times New Roman.'.format(length),
            ' Your answer should fit onto the front page of my newspaper and should therefore be exactly {:.2f} cms of 12pt Times New Roman long.'.format(length),
            ' Please generate text of {:.2f} cms length, assuming 12pt Times New Roman is used as font.'.format(length),
            ' Your response should be {:.2f} cms long if printed out in 12pt Times New Roman.'.format(length),
            ' I am going to print out your reply using 12pt Times New Roman and need you reply to be {:.2f} cms long.'.format(length),
            ]

        return question + random.choice(strs)
    return question


def load_results(dir_path):
    #utility to load all result df in a folder at once, assumes folder only contains pandas dataframes stored with pickle
    file_names = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    data_frames = []
    for file_name in file_names:
        if file_name[-3:] == ".df":
            with open(dir_path+"/"+file_name, 'rb') as f:
                data_frames.append((file_name, pickle.load(f)))
    return data_frames

def split_messages(messages, ignore_system=True):
    #uses regex to split raw output of Llama 3.1 into list of messages assigned to system, user or assistant
    strs = re.split(r"<\|eot_id\|>", messages)
    result = []
    for str in strs:
        if "<|start_header_id|>system<|end_header_id|>" in str and not ignore_system:
            result.append(str[str.find('<|end_header_id|>')+17:].rstrip().lstrip())
        if "<|start_header_id|>user<|end_header_id|>" in str:
            result.append(str[str.find('<|end_header_id|>')+17:].rstrip().lstrip())
        if "<|start_header_id|>assistant<|end_header_id|>" in str:
            result.append(str[str.find('<|end_header_id|>')+17:].rstrip().lstrip())
    return result

def create_base_llama_responses(dir_path, batch_size=32):
    #this function samples baseline Llama 3.1 to generate responses to questions in a result df
    #this is useful for calculating semscore/bertscore

    #IMPORTANT: THIS FUNCTION ASSUMES THAT ALL RESULTS IN THE DIRECTORY USED THE SAME BASE QUESTIONS
    from transformers import pipeline
    from datasets import Dataset
    import torch
    from transformers.pipelines.pt_utils import KeyDataset
    #importing here looks kinda scuffed but improves performance when calling other functions from this file

    data_list = load_results(dir_path)
    save_path = dir_path + "/baseline_llama_responses"
    pipe = pipeline(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=2048,
                        model_kwargs={"torch_dtype": torch.bfloat16}, device_map = "auto")
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    baseline_responses = []
    question_data = Dataset.from_dict({"input" : data_list[0][1]["base_question"].to_list()})

    for i, output in enumerate(pipe(KeyDataset(question_data, "input"), batch_size=batch_size, truncation="only_first")):
        output_str = output[0]["generated_text"]
        baseline_responses.append(output_str.replace(question_data["input"][i],""))

    with open(save_path , mode="wb") as f:
        pickle.dump(baseline_responses, file=f)


def compare_responses_to_base_llama(dir_path, save_path=None, base_responses_path = None, base_responses_is_df=False):
    from utils.get_sem_score import get_sem_score
    '''
    This function generates semscore+bertscore between the responses stored in dataframes in one folder and a dataframe
    containing the baseline responses. Such a dataframe can easily be produced using the create_base_llama_responses 
    function.
    '''

    if save_path is None:
        save_path = dir_path
    if base_responses_path is None:
        base_responses_path = dir_path + "/baseline_llama_responses"
    with open(base_responses_path, 'rb') as f:
        baseline_responses = pickle.load(f)
    data_list = load_results(dir_path)
    result_list = []

    for data in tqdm(data_list):
        df = data[1]
        if not "bert_score_llama_response" in df and not "sem_score_llama_response" in df:
            sem_score_list = []
            response_list = []
            for i, sample in df.iterrows():
                messages = sample["messages"]
                response = split_messages(messages)[-1]
                if base_responses_is_df:
                    baseline_response = None
                else:
                    baseline_response = baseline_responses[i]
                recall, precision, f1 = get_sem_score(response, baseline_response)
                sem_score = {"recall": recall, "precision": precision, "f1": f1}
                sem_score_list.append(sem_score)
                response_list.append(response)
            bert_score_dict = get_bert_score(response_list, baseline_responses)
            bert_score_list = []
            for i in range(len(bert_score_dict["recall"])):
                bert_score_list.append({"recall": bert_score_dict["recall"][i], "precision": bert_score_dict["precision"][i],"f1": bert_score_dict["f1"][i]})
            result_list.append({"sem_scores": sem_score_list, "bert_scores": bert_score_list})
            df.insert(0, "bert_score_llama_response", bert_score_list)
            df.insert(0, "sem_score_llama_response", sem_score_list)
            with open(save_path+"/"+data[0], mode="wb") as f:
                pickle.dump(df, file=f)



class BestOfNModelWrapper:
    def __init__(self, pipe, temp_min, temp_max, n, use_custom_llama=False):
        '''
        Wrapper class to allow a pipe or a custom Llama object (= Wrapper for Llama with changing output distribution
        based on length, see custom_llama_3_1.py) to pick the best result based on n generations.
        '''
        self.pipe = pipe
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.use_custom_llama = use_custom_llama
        self.n = n


    def generate(self, query, target, eval_mode, language):
        #this is slow and shouldn't be used
        response_list = []
        diff_list = []
        for i in range(self.n):
            if self.use_custom_llama:
                pred = self.pipe.generate_from_prompt(query, target, eval_mode, language, temp=random.uniform(self.temp_min, self.temp_max))[0]
                output_str = query + "\n" + pred
            else:
                output_str = self.pipe(query, temperature=random.uniform(self.temp_min, self.temp_max))[0]["generated_text"]
                pred = output_str.replace(query,"")
            length = get_length(pred, eval_mode=eval_mode, language=language)
            diff = abs(length-target)
            response_list.append((output_str, pred))
            diff_list.append(diff)
        idx_best = diff_list.index(min(diff_list))
        return response_list[idx_best]

    def batched_generate(self, batch, batch_size, language):
        response_list = []
        diff_list = []
        for j in range(self.n):
            for i, output in enumerate(self.pipe(KeyDataset(batch, "input"), batch_size=batch_size, truncation="only_first", temperature=random.uniform(self.temp_min, self.temp_max))):
                output_str = output[0]["generated_text"]
                pred = output_str.replace(batch["input"][i], "")
                target = batch["length"][i]
                eval_mode = batch["eval_mode"][i]
                length = get_length(pred, eval_mode=eval_mode, language=language)
                diff = abs(length-target)
                if j == 0:
                    response_list.append([(output_str, pred)])
                    diff_list.append([diff])
                else:
                    response_list[i].append((output_str, pred))
                    diff_list[i].append(diff)
        best_output_list = []
        best_pred_list = []
        for i in range(batch_size):
            idx_best = diff_list[i].index(min(diff_list[i]))
            best_output_list.append(response_list[i][idx_best][0])
            best_pred_list.append(response_list[i][idx_best][1])
        return best_output_list, best_pred_list

def dynamic_length_generation(pipe, base_tokenizer, prompt, target_len, eval_mode, language, temperature=.5, max_tokens_between_target_change=5, min_tokens_between_target_change=1, min_remaining_length=.25, length_change_scale=1):
    # this function implements generation with a dynamically changing length requirement
    pred = ""
    curr_target_len = target_len
    while True:
        question_str = format_question(prompt, eval_mode, curr_target_len)
        question_str = base_tokenizer.apply_chat_template([{"role": "user", "content": question_str}], tokenize=False, add_generation_prompt=True)
        input_str = question_str +  pred
        output_str = pipe(input_str, max_new_tokens=random.randint(min_tokens_between_target_change,max_tokens_between_target_change), temperature=temperature, pad_token_id=pipe.tokenizer.eos_token_id)[0]["generated_text"]
        pred = output_str.replace(question_str, "")
        if input_str == output_str:
            break
        length = get_length(pred, eval_mode, language)
        remaining_len = target_len - length
        curr_target_len = target_len + remaining_len * np.random.normal(loc=.0, scale=length_change_scale)
        curr_target_len = max(curr_target_len, length*(1+min_remaining_length))
        curr_target_len = min(curr_target_len, target_len + remaining_len*(1-min_remaining_length))

        if length > target_len:
            question_str = format_question(prompt, eval_mode, curr_target_len)
            question_str = base_tokenizer.apply_chat_template([{"role": "user", "content": question_str}], tokenize=False, add_generation_prompt=True)
            input_str = question_str + pred
            output_str = pipe(input_str, max_new_tokens=2048, temperature=temperature, pad_token_id=pipe.tokenizer.eos_token_id)[0]["generated_text"]
            pred = output_str.replace(question_str, "")
            break


    return pred, output_str, curr_target_len




