from utils.eval_utils import split_messages
import numpy as np
import statistics



def extract_len_diff_data(data_list):
    col_chars_list = []
    col_letters_list = []
    col_spoken_list = []
    col_space_list = []
    for data in data_list:
        data_chars = data.loc[data['mode'] == "chars"]
        data_letters = data.loc[data['mode'] == "letters"]
        data_spoken = data.loc[data['mode'] == "spoken"]
        data_space = data.loc[data['mode'] == "space"]
        col_chars_list.append(np.array(data_chars["len_actual"]) - np.array(data_chars["len_target"]))
        col_letters_list.append(np.array(data_letters["len_actual"]) - np.array(data_letters["len_target"]))
        col_spoken_list.append(np.array(data_spoken["len_actual"]) - np.array(data_spoken["len_target"]))
        col_space_list.append(np.array(data_space["len_actual"]) - np.array(data_space["len_target"]))
    return col_chars_list, col_letters_list, col_spoken_list, col_space_list

def extract_len_diff_words(data_list):
    col_words_list = []
    for data in data_list:
        data_words = data.loc[data['mode'] == "words"]
        col_words_list.append(np.array(data_words["len_actual"]) - np.array(data_words["len_target"]))
    return col_words_list

def extract_len_diff_dyn_len(data_list):
    col_spoken_list = []
    percent_diff_list = []
    for data in data_list:
        data_spoken = data.loc[data['mode'] == "spoken"]
        col_spoken_list.append(np.array(data_spoken["len_actual"]) - np.array(data_spoken["len_target_end"]))
        percent_diff_list.append((np.array(data_spoken["len_actual"]) - np.array(data_spoken["len_target_end"]))/np.array(data_spoken["len_target_end"]))
    return col_spoken_list, percent_diff_list

def extract_mean_len_diff_data(data_list):
    mean_diff_chars_list = []
    mean_diff_letters_list = []
    mean_diff_spoken_list = []
    mean_diff_space_list = []
    for data in data_list:
        data_chars = data.loc[data['mode'] == "chars"]
        data_letters = data.loc[data['mode'] == "letters"]
        data_spoken = data.loc[data['mode'] == "spoken"]
        data_space = data.loc[data['mode'] == "space"]
        mean_diff_chars_list.append(data_chars.loc[:, 'len_diff'].mean())
        mean_diff_letters_list.append(data_letters.loc[:, 'len_diff'].mean())
        mean_diff_spoken_list.append(data_spoken.loc[:, 'len_diff'].mean())
        mean_diff_space_list.append(data_space.loc[:, 'len_diff'].mean())
    return mean_diff_chars_list, mean_diff_letters_list, mean_diff_spoken_list, mean_diff_space_list

def extract_mean_bert_sem_score_data(data_list):
    sem_score_gpt_list = []
    bert_score_gpt_list = []
    sem_score_llama_list = []
    bert_score_llama_list = []
    for data in data_list:
        sem_score_gpt = []
        bert_score_gpt = []
        sem_score_llama = []
        bert_score_llama = []
        for i, sample in data.iterrows():
            sem_score_gpt.append(sample["sem_score_gpt_response"]["f1"])
            bert_score_gpt.append(sample["bert_score_gpt_response"]["f1"])
            sem_score_llama.append(sample["sem_score_llama_response"]["f1"])
            bert_score_llama.append(sample["bert_score_llama_response"]["f1"])
        sem_score_gpt_list.append(statistics.fmean(sem_score_gpt))
        bert_score_gpt_list.append(statistics.fmean(bert_score_gpt))
        sem_score_llama_list.append(statistics.fmean(sem_score_llama))
        bert_score_llama_list.append(statistics.fmean(bert_score_llama))
    return sem_score_gpt_list, bert_score_gpt_list, sem_score_llama_list, bert_score_llama_list

def extract_bert_sem_score_data(data_list):
    sem_score_gpt_list = []
    bert_score_gpt_list = []
    sem_score_llama_list = []
    bert_score_llama_list = []
    for data in data_list:
        sem_score_gpt = []
        bert_score_gpt = []
        sem_score_llama = []
        bert_score_llama = []
        for i, sample in data.iterrows():
            sem_score_gpt.append(sample["sem_score_gpt_response"]["f1"])
            bert_score_gpt.append(sample["bert_score_gpt_response"]["f1"])
            sem_score_llama.append(sample["sem_score_llama_response"]["f1"])
            bert_score_llama.append(sample["bert_score_llama_response"]["f1"])
        sem_score_gpt_list.append(np.array(sem_score_gpt))
        bert_score_gpt_list.append(np.array(bert_score_gpt))
        sem_score_llama_list.append(np.array(sem_score_llama))
        bert_score_llama_list.append(np.array(bert_score_llama))
    return sem_score_gpt_list, bert_score_gpt_list, sem_score_llama_list, bert_score_llama_list


def extract_grammar_error_data(data_list, base_dataset):
    grammar_error_ref_list = []
    grammar_error_pred_list = []
    for data in data_list:
        grammar_error_ref = []
        grammar_error_pred = []

        for i, sample in data.iterrows():
            pred_response_without_tokens = split_messages(sample["messages"])[-1]
            pred_length_in_chars = len(pred_response_without_tokens)
            pred_grammar_err_count = len(sample["pred_grammar_errors"])
            pred_normalized_error_count = pred_grammar_err_count / pred_length_in_chars
            grammar_error_pred.append(pred_normalized_error_count)

            ref_response_without_tokens = base_dataset[i]["first_turn"].replace(base_dataset[i]["question"], "" ).lstrip()
            ref_length_in_chars = len(ref_response_without_tokens)
            ref_grammar_err_count = len(sample["ref_grammar_errors"])
            ref_normalized_error_count = ref_grammar_err_count / ref_length_in_chars
            grammar_error_ref.append(ref_normalized_error_count)
        grammar_error_ref_list.append(np.array(grammar_error_ref))
        grammar_error_pred_list.append(np.array(grammar_error_pred))
    return grammar_error_pred_list, grammar_error_ref_list

def extract_mean_grammar_error_data(data_list, base_dataset):
    grammar_error_ref_list = []
    grammar_error_pred_list = []
    for data in data_list:
        grammar_error_ref = []
        grammar_error_pred = []

        for i, sample in data.iterrows():
            pred_response_without_tokens = split_messages(sample["messages"])[-1]
            pred_length_in_chars = len(pred_response_without_tokens)
            pred_grammar_err_count = len(sample["pred_grammar_errors"])
            pred_normalized_error_count = pred_grammar_err_count / pred_length_in_chars
            grammar_error_pred.append(pred_normalized_error_count)

            ref_response_without_tokens = base_dataset[i]["first_turn"].replace(base_dataset[i]["question"], "" ).lstrip()
            ref_length_in_chars = len(ref_response_without_tokens)
            ref_grammar_err_count = len(sample["ref_grammar_errors"])
            ref_normalized_error_count = ref_grammar_err_count / ref_length_in_chars
            grammar_error_ref.append(ref_normalized_error_count)
        grammar_error_ref_list.append(statistics.fmean(grammar_error_ref))
        grammar_error_pred_list.append(statistics.fmean(grammar_error_pred))
    return grammar_error_pred_list, grammar_error_ref_list