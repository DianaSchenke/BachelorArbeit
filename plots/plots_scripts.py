from plot_table_production import *
from data_extraction import *
from utils.eval_utils import load_results
from utils.dataset_utils import DatasetWrapper
import pickle

#this file contains various scripts that make plots for my thesis by extracting data from pandas dataframes and then using matplotlib

def sft_overfit_results():
    path = "/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_sft"
    result_list = load_results(path)
    result_list.sort()
    result_list[0], result_list[-1] = result_list[-1], result_list[0]
    name_list = []
    data_list = []
    for result in result_list:
        name_list.append(result[0].replace("_epochs_sft.df", "").replace("erence.df", ""))
        data_list.append(result[1])
    name_list[0] = 0

    data_wrapper = DatasetWrapper("/data/home/scl33452/PycharmProjects/Training-Setup/base_datasets/questions_about_the_world")
    data_wrapper.shuffle(42)
    samples = 1280
    base_dataset = data_wrapper.get_eval_split(samples=samples)

    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    grammar_error_pred_list, grammar_error_ref_list = extract_grammar_error_data(data_list, base_dataset)
    mean_sem_score_gpt_list, _, mean_sem_score_llama_list, _ = extract_mean_bert_sem_score_data(data_list)

    produce_rmse_len_diff_table(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, title="", index_col_name="Epochs", index_col_format="X", filename="sft_overfit_len_diff_table.txt", decimals=2)
    produce_sem_score_table(mean_sem_score_gpt_list, mean_sem_score_llama_list, name_list, title="", index_col_name="Epochs", index_col_format="X", filename="sft_overfit_semscore_table.txt", decimals=2)
    produce_error_table(grammar_error_pred_list+[grammar_error_ref_list[0]], name_list+["GPT"],title="", index_col_name="Epochs", index_col_format="X", filename="sft_overfit_error_table.txt", decimals=2)
    produce_len_diff_plot(col_chars_list[1:], col_letters_list[1:], col_spoken_list[1:], col_space_list[1:], name_list[1:], filename="sft_overfit_len_diff_no_base.png")


def rlhf_method_comparison_results():
    data_list = []

    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_sft/3_epochs_sft.df", 'rb') as f:
        data_list.append(pickle.load(f))

    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_rlhf/dpo.df", 'rb') as f:
        data_list.append(pickle.load(f))

    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_rlhf/orpo.df", 'rb') as f:
        data_list.append(pickle.load(f))

    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_rlhf/ppo.df", 'rb') as f:
        data_list.append(pickle.load(f))

    name_list = ["SFT", "DPO", "ORPO", "PPO"]

    data_wrapper = DatasetWrapper("/data/home/scl33452/PycharmProjects/Training-Setup/base_datasets/questions_about_the_world")
    data_wrapper.shuffle(42)
    samples = 1280
    base_dataset = data_wrapper.get_eval_split(samples=samples)

    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    grammar_error_pred_list, grammar_error_ref_list = extract_grammar_error_data(data_list, base_dataset)
    mean_sem_score_gpt_list, _, mean_sem_score_llama_list, _ = extract_mean_bert_sem_score_data(data_list)


    produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="rlhf_method_comparison_len_diff.png", plot_outliers=True, x_label="Fine-Tuning Method")
    produce_sem_score_table(mean_sem_score_gpt_list, mean_sem_score_llama_list, name_list,
                            title="", index_col_name="model", index_col_format="X", filename="rlhf_method_comparison_semscore_table.txt", decimals=2)

    produce_error_table(grammar_error_pred_list, name_list,title="", index_col_name="model", index_col_format="X", filename="rlhf_method_comparison_error_table.txt", decimals=2)

def best_model_comparison_results():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_llama_responses_comparison/trained_on_llama.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_llama_responses_comparison/trained_on_gpt.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_llama_rlhf/orpo.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_rlhf/orpo.df", 'rb') as f:
        data_list.append(pickle.load(f))

    name_list = ["Llama", "GPT", "Llama", "GPT"]

    data_wrapper = DatasetWrapper("/data/home/scl33452/PycharmProjects/Training-Setup/base_datasets/questions_about_the_world")
    data_wrapper.shuffle(42)
    samples = 1280
    base_dataset = data_wrapper.get_eval_split(samples=samples)

    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    mean_sem_score_gpt_list, _, mean_sem_score_llama_list, _ = extract_mean_bert_sem_score_data(data_list)
    grammar_error_pred_list, grammar_error_ref_list = extract_grammar_error_data(data_list, base_dataset)

    produce_best_model_comparison_plot_len_diff(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="best_model_comparison_len_diff.png", x_label="Training Data")
    produce_sem_score_table(mean_sem_score_gpt_list, mean_sem_score_llama_list, name_list,
                            title="", index_col_name="model", index_col_format="X", filename="best_model_comparison_semscore_table.txt", decimals=2)
    produce_error_table(grammar_error_pred_list, name_list,title="", index_col_name="model", index_col_format="X", filename="best_model_comparison_error_table.txt", decimals=2)


def dataset_test_wac_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/wac_test/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/wac_test/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))
    name_list = ["Base Model", "Trained Model"]
    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="dataset_test_wac_len_diff.png", x_label="")

def dataset_test_aem_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/aem_test/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/aem_test/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))
    name_list = ["Base Model", "Trained Model"]
    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="dataset_test_aem_len_diff.png", x_label="")

def word_requirement_test_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_words_requirement/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_words_requirement/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))
    name_list = ["Base Model", "Trained Model"]
    col_words_list = extract_len_diff_words(data_list)
    produce_len_diff_plot_words(col_words_list, name_list, filename="word_requirement_test_len_diff.png", x_label="")

def question_strs_test_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/other_question_strs/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/other_question_strs/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))
    name_list = ["Base Model", "Trained Model"]
    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="question_strs_test_len_diff.png", x_label="")

def dyn_len_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/dyn_len/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/dyn_len/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))
    name_list = ["Base Model", "Trained Model"]
    col_spoken_list, percent_diff_list = extract_len_diff_dyn_len(data_list)
    produce_len_diff_plot_spoken_only(col_spoken_list, percent_diff_list, name_list, filename="dyn_len_test_len_diff.png", x_label="")
    produce_abstract_percent_in_quantile_table(percent_diff_list, name_list, title="", index_col_name="", index_col_format="X", filename="abstract_dyn_len_table.txt", decimals=3)

def comparison_for_abstract():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_rlhf/orpo.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_sft/reference.df", 'rb') as f:
        data_list.append(pickle.load(f))

    name_list = ["our model", "baseline"]
    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="abstract_model_comparison.png", plot_outliers=True, x_label="")
    produce_rmse_len_diff_table(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, title="", index_col_name="", index_col_format="X", filename="abstract_rmse_table.txt", decimals=2)

def naive_methods_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/qaw_rlhf/orpo.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/naive_methods/best_of_2.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/naive_methods/eos_prob.df", 'rb') as f:
        data_list.append(pickle.load(f))
    #with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/naive_methods/best_of_2_and_eos_prob.df", 'rb') as f:
    #    data_list.append(pickle.load(f))

    name_list = ["Baseline", "Best of 2", "Token Probs"]#, "both"]
    col_chars_list, col_letters_list, col_spoken_list, col_space_list = extract_len_diff_data(data_list)
    produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename="naive_methods_len_diff.png", plot_outliers=True, x_label="")

def conversation_test_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/conversation_test/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/conversation_test/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))
    name_list = ["Base Model","Trained Model"]
    len_list = []
    for df in data_list:
        len_list.append(df["len_actual"])
    produce_len_plot_chars(len_list, name_list, filename="conversation_test_len_diff.png", x_label="", h_line=400)

def article_summaries_test_len_diff():
    data_list = []
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/article_summary_test/base.df", 'rb') as f:
        data_list.append(pickle.load(f))
    with open("/data/home/scl33452/PycharmProjects/Training-Setup/evaluation_scripts/results/article_summary_test/trained.df", 'rb') as f:
        data_list.append(pickle.load(f))

    name_list = ["Base Model","Trained Model"]
    len_list = []
    for df in data_list:
        len_list.append(df["len_actual"])


    produce_len_plot_space(len_list, name_list, filename="article_summaries_test_len_diff.png", x_label="", h_line=30)


import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

sft_overfit_results()
rlhf_method_comparison_results()
best_model_comparison_results()
dataset_test_wac_len_diff()
dataset_test_aem_len_diff()
word_requirement_test_len_diff()
question_strs_test_len_diff()
dyn_len_len_diff()
comparison_for_abstract()
naive_methods_len_diff()
conversation_test_len_diff()
article_summaries_test_len_diff()

