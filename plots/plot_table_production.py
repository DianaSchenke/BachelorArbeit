from utils.plot_utils import make_pretty_box_plot, to_tabularx, make_pretty_violin_plot, make_pretty_violin_plot_with_outliers, make_split_violin_plot_with_outliers
import numpy as np
import pandas as pd



def produce_len_diff_plot(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename, x_label="Training Epochs", plot_outliers=True):
    data = [
        {
            "data" : col_chars_list,
            "x_label" : x_label,
            "y_label" : "Deviation (Characters)",
            "title" : "Character Count Length Requirement",
            "tick_labels" : name_list,
        },
        {
            "data": col_letters_list,
            "x_label": x_label,
            "y_label": "Deviation (Letters)",
            "title": "Letter Count Length Requirement",
            "tick_labels": name_list,
        },
        {
            "data": col_spoken_list,
            "x_label": x_label,
            "y_label": "Deviation (Seconds)",
            "title": "Speech Length Requirement",
            "tick_labels": name_list,
        },
        {
            "data": col_space_list,
            "x_label": x_label,
            "y_label": "Deviation (cm)",
            "title": "Print Length Requirement",
            "tick_labels": name_list,
        },
    ]
    make_pretty_violin_plot_with_outliers(data, "", filename)

def produce_len_diff_plot_spoken_only(col_spoken_list, percent_diff_list, name_list, filename, x_label="Training Epochs", plot_outliers=True):
    data = [
        #{
        #    "data": col_spoken_list,
        #    "x_label": x_label,
        #    "y_label": "deviation (seconds)",
        #    "title": "final speech length requirement",
        #    "tick_labels": name_list,
        #},
        {
            "data": percent_diff_list,
            "x_label": x_label,
            "y_label": "Deviation From Final Length Requirement",
            "title": "",
            "tick_labels": name_list,
            "format_percent": True,
        },
    ]
    make_pretty_box_plot(data, "", filename, showfliers=False)

def produce_len_plot_chars(col_chars_list, name_list, filename, x_label="Training Epochs", plot_outliers=True, h_line=None):
    data = [
        {
            "data" : col_chars_list,
            "x_label" : x_label,
            "y_label" : "Length (Characters)",
            "title" : "",
            "tick_labels" : name_list,
        },
    ]
    make_pretty_violin_plot_with_outliers(data, "", filename, h_line=h_line)

def produce_len_plot_space(col_space_list, name_list, filename, x_label="Training Epochs", plot_outliers=True, h_line=None):
    data = [
        {
            "data": col_space_list,
            "x_label": x_label,
            "y_label": "Length (cm)",
            "title": "",
            "tick_labels": name_list,
        },
    ]
    make_pretty_violin_plot_with_outliers(data, "", filename, h_line=h_line)

def produce_len_diff_plot_words(col_words_list, name_list, filename, x_label="Training Epochs"):
    data = [
        {
            "data": col_words_list,
            "x_label": x_label,
            "y_label": "Deviation (Words)",
            "title": "",
            "tick_labels": name_list,
        },
    ]
    make_pretty_violin_plot_with_outliers(data, "", filename)

def produce_sem_bert_score_plot(sem_score_gpt_list, bert_score_gpt_list, sem_score_llama_list, bert_score_llama_list, filename, name_list, x_label="Training Epochs"):
    data = [
        {
            "data" : sem_score_gpt_list,
            "x_label" : x_label,
            "y_label" : "SemScore",
            "title" : "Similarity to GPT Generated Responses (SemScore)",
            "tick_labels" : name_list,
        },
        {
            "data": bert_score_gpt_list,
            "x_label": x_label,
            "y_label": "BERTScore",
            "title": "Similarity to GPT generated responses (BERTScore)",
            "tick_labels": name_list,
        },
        {
            "data": sem_score_llama_list,
            "x_label": x_label,
            "y_label": "SemScore",
            "title": "Similarity to Llama generated responses (SemScore)",
            "tick_labels": name_list,
        },
        {
            "data": bert_score_llama_list,
            "x_label": x_label,
            "y_label": "BERTScore",
            "title": "Similarity to Llama generated responses (BERTScore)",
            "tick_labels": name_list,
        },
    ]

    make_pretty_violin_plot(data, title="", filename=filename)

def produce_best_model_comparison_plot_len_diff(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, filename, x_label="Training Epochs"):
    data_left = [
        {
            "data" : col_chars_list[:2],
            "x_label" : x_label,
            "y_label": "Deviation (Characters)",
            "title": "Character Count Length Requirement",
            "tick_labels" : name_list[:2],
        },
        {
            "data": col_letters_list[:2],
            "x_label": x_label,
            "y_label": "Deviation (Letters)",
            "title": "Letter Count Length Requirement",
            "tick_labels": name_list[:2],
        },
        {
            "data": col_spoken_list[:2],
            "x_label": x_label,
            "y_label": "Deviation (Seconds)",
            "title": "Speech Length Requirement",
            "tick_labels": name_list[:2],
        },
        {
            "data": col_space_list[:2],
            "x_label": x_label,
            "y_label": "Deviation (cm)",
            "title": "Print Length Requirement",
            "tick_labels": name_list[:2],
        },
    ]
    data_right = [
        {
            "data" : col_chars_list[2:],
            "x_label" : x_label,
            "y_label": "Deviation (Characters)",
            "title": "Character Count Length Requirement",
            "tick_labels" : name_list[2:],
        },
        {
            "data": col_letters_list[2:],
            "x_label": x_label,
            "y_label": "Deviation (Letters)",
            "title": "Letter Count Length Requirement",
            "tick_labels": name_list[2:],
        },
        {
            "data": col_spoken_list[2:],
            "x_label": x_label,
            "y_label": "Deviation (Seconds)",
            "title": "Speech Length Requirement",
            "tick_labels": name_list[2:],
        },
        {
            "data": col_space_list[2:],
            "x_label": x_label,
            "y_label": "Deviation (cm)",
            "title": "Print Length Requirement",
            "tick_labels": name_list[2:],
        },
    ]
    make_split_violin_plot_with_outliers(data_left, data_right, left_name="Llama Test Data", right_name="GPT Test Data", title="", filename=filename)

def produce_model_comparison_bert_sem_score_plot(sem_score_gpt_list, bert_score_gpt_list, sem_score_llama_list, bert_score_llama_list, filename, name_list, x_label="Training Epochs"):
    data_left = [
        {
            "data" : sem_score_gpt_list[:2],
            "x_label" : x_label,
            "y_label" : "SemScore",
            "title" : "Similarity to GPT generated responses (SemScore)",
            "tick_labels" : name_list[:2],
        },
        {
            "data": bert_score_gpt_list[:2],
            "x_label": x_label,
            "y_label": "BERTScore",
            "title": "Similarity to GPT generated responses (BERTScore)",
            "tick_labels": name_list[:2],
        },
        {
            "data": sem_score_llama_list[:2],
            "x_label": x_label,
            "y_label": "SemScore",
            "title": "Similarity to Llama generated responses (SemScore)",
            "tick_labels": name_list[:2],
        },
        {
            "data": bert_score_llama_list[:2],
            "x_label": x_label,
            "y_label": "BERTScore",
            "title": "Similarity to Llama generated responses (BERTScore)",
            "tick_labels": name_list[:2],
        },
    ]
    data_right = [
        {
            "data" : sem_score_gpt_list[2:],
            "x_label" : x_label,
            "y_label" : "SemScore",
            "title" : "Similarity to GPT generated responses (SemScore)",
            "tick_labels" : name_list[2:],
        },
        {
            "data": bert_score_gpt_list[2:],
            "x_label": x_label,
            "y_label": "BERTScore",
            "title": "Similarity to GPT generated responses (BERTScore)",
            "tick_labels": name_list[2:],
        },
        {
            "data": sem_score_llama_list[2:],
            "x_label": x_label,
            "y_label": "SemScore",
            "title": "Similarity to Llama generated responses (SemScore)",
            "tick_labels": name_list[2:],
        },
        {
            "data": bert_score_llama_list[2:],
            "x_label": x_label,
            "y_label": "BERTScore",
            "title": "Similarity to Llama generated responses (BERTScore)",
            "tick_labels": name_list[2:],
        },
    ]
    make_split_violin_plot_with_outliers(data_left, data_right, left_name="Llama test data", right_name="GPT test data", title="", filename=filename)

def produce_abstract_table(data_array, name_list, title, index_col_name, index_col_format, filename, decimals=2, rmse=True):
        quartile1, medians, quartile3 = np.percentile(data_array, [25, 50, 75], axis=1)
        col_format = f"|{index_col_format}|X|X|X|X|"
        result_dict = {
            index_col_name: name_list,
            "1st Quartile": quartile1,
            "Median": medians,
            "3rd Quartile": quartile3,
        }
        if rmse:
            result_dict["RMSE"] = np.sqrt(np.average(data_array**2, axis=1))
        caption = ""
        with open(filename, "w") as text_file:
            text_file.write(to_tabularx(pd.DataFrame.from_dict(result_dict), decimals, caption=caption, title=title,col_format=col_format))

def produce_rmse_len_diff_table(col_chars_list, col_letters_list, col_spoken_list, col_space_list, name_list, title, index_col_name, index_col_format, filename, decimals=2):
    result_dict = {
        index_col_name: name_list,
        "RMSE (characters)": np.sqrt(np.average(np.array(col_chars_list)**2, axis=1)),
        "RMSE (letters)": np.sqrt(np.average(np.array(col_letters_list)**2, axis=1)),
        "RMSE (spoken)": np.sqrt(np.average(np.array(col_spoken_list)**2, axis=1)),
        "RMSE (printed)": np.sqrt(np.average(np.array(col_space_list)**2, axis=1)),
    }
    caption = ""
    col_format = f"|{index_col_format}|X|X|X|X|"
    df = pd.DataFrame.from_dict(result_dict)
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(df, decimals, caption=caption, title=title, col_format=col_format, index=False))

def produce_abstract_percent_in_quantile_table(percent_diff_list, name_list, title, index_col_name, index_col_format, filename, decimals=2):
    five_percent = []
    ten_percent = []
    twenty_five_percent = []
    fifty_percent = []
    hundred_percent = []
    two_fifty_percent = []
    for arr in percent_diff_list:
        arr = np.abs(arr)
        five_percent.append(len(arr[arr<.05])/len(arr))
        ten_percent.append(len(arr[arr<.1])/len(arr))
        twenty_five_percent.append(len(arr[arr<.25])/len(arr))
        fifty_percent.append(len(arr[arr<.5])/len(arr))
        hundred_percent.append(len(arr[arr<1])/len(arr))
        two_fifty_percent.append(len(arr[arr<2.5])/len(arr))

    result_dict = {
        index_col_name: name_list,
        "within 5%": five_percent,
        "within 10%": ten_percent,
        "within 25%": twenty_five_percent,
        "within 50%": fifty_percent,
        "within 100%": hundred_percent,
        "within 250%": two_fifty_percent,
    }
    caption = ""
    col_format = f"|{index_col_format}|X|X|X|X|"
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(pd.DataFrame.from_dict(result_dict), decimals, caption=caption, title=title, col_format=col_format))


def produce_grammar_error_plot(grammar_error_pred_list, grammar_error_ref_list, filename, name_list, x_label="Training Epochs"):
    name_list.append("gpt")
    normalized_error_counts = []
    for x in grammar_error_pred_list:
        normalized_error_counts.append(np.array(x) * 1000)
    normalized_error_counts.append(np.array(grammar_error_ref_list[0])*1000)
    data = [
        {
            "data" : normalized_error_counts,
            "x_label" : x_label,
            "y_label" : "Errors per 1000 Characters",
            "title" : "",
            "tick_labels" : name_list,
        },
    ]
    make_pretty_box_plot(data, title="", filename=filename, showfliers=False)

def produce_grammar_error_table(grammar_error_pred_list, grammar_error_ref_list, name_list, title, index_col_name, index_col_format, grammar_err_pred_col_name, filename, decimals=2):
    col_format = f"|{index_col_format}|X|X|"
    result_dict = {
        index_col_name : name_list,
        "Grammar Errors (per 1000 Characters)": np.array(grammar_error_pred_list) * 1000,
        grammar_err_pred_col_name: np.array(grammar_error_pred_list) / np.array(
            grammar_error_ref_list)
    }
    caption=""
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(pd.DataFrame.from_dict(result_dict), decimals, caption=caption, title=title, col_format=col_format))


def produce_bert_sem_score_table(sem_score_gpt_list, bert_score_gpt_list, sem_score_llama_list, bert_score_llama_list,
                                 name_list, title, index_col_name, index_col_format, filename, decimals=2):

    col_format = f"|{index_col_format}|X|X|X|X|"
    result_dict = {
        index_col_name : name_list,
        "\\(\\text{SemScore}_{\\text{GPT}}\\)": sem_score_gpt_list,
        "\\(\\text{BertScore}_{\\text{GPT}}\\)": bert_score_gpt_list,
        "\\(\\text{SemScore}_{\\text{Llama}}\\)": sem_score_llama_list,
        "\\(\\text{BertScore}_{\\text{Llama}}\\)": bert_score_llama_list,
    }
    caption=""
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(pd.DataFrame.from_dict(result_dict), decimals, caption=caption, title=title, col_format=col_format))

def produce_sem_score_table(sem_score_gpt_list, sem_score_llama_list, name_list,title, index_col_name, index_col_format, filename, decimals=2):
    result_dict = {
        index_col_name: name_list,
        "GPT": sem_score_gpt_list,
        "Llama":sem_score_llama_list,
    }
    caption=""
    df = pd.DataFrame.from_dict(result_dict)
    df = df.T
    col_format = f"|{index_col_format}|" + "|X"*len(sem_score_gpt_list)
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(df, decimals, caption=caption, title=title, col_format=col_format, header=False, index=True))

def produce_error_table(grammar_error_pred_list, name_list, title, index_col_name, index_col_format, filename, decimals=2):

    result_dict = {
        index_col_name: name_list,
        "grammar errors (per 1000 characters)": np.mean(np.array(grammar_error_pred_list) * 1000, 1),
    }
    caption=""
    df = pd.DataFrame.from_dict(result_dict)
    df = df.T
    col_format = f"|{index_col_format}|" + "|X"*len(grammar_error_pred_list)
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(df, decimals, caption=caption, title=title, col_format=col_format, header=False, index=True))



def produce_mean_len_diff_table(mean_diff_chars_list, mean_diff_letters_list, mean_diff_spoken_final, mean_diff_space_final, name_list, title, index_col_name, index_col_format, filename, decimals=2):
    col_format = f"|{index_col_format}|X|X|X|X|"
    result_dict = {
        index_col_name : name_list,
        "\\(\\Delta_{\\text{chars}}\\)": mean_diff_chars_list,
        "\\(\\Delta_{\\text{letters}}\\)": mean_diff_letters_list,
        "\\(\\Delta_{\\text{spoken}}\\)": mean_diff_spoken_final,
        "\\(\\Delta_{\\text{printed}}\\)": mean_diff_space_final,
    }
    caption = ("\\(\\Delta_{\\text{chars}}\\), \\(\\Delta_{\\text{letters}}\\), \\(\\Delta_{\\text{spoken}}\\) and \\(\\Delta_{\\text{printed}}\\) are the mean difference between the length of the generated text and the length requirement, "
               "with the requirement stated as number of characters, number of letters, time to read text aloud in seconds and length of text in cm if printed in 12pt Times New Roman respectively.")
    with open(filename, "w") as text_file:
        text_file.write(to_tabularx(pd.DataFrame.from_dict(result_dict), decimals, caption=caption, title=title, col_format=col_format))