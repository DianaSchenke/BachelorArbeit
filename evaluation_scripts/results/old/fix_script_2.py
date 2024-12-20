from utils.dataset_utils import DatasetWrapper
import pathlib
import os
from utils.eval_utils import load_results, split_messages, get_bert_score
from utils.get_sem_score import get_sem_score
import pickle
from tqdm import tqdm

base_dir = str(pathlib.Path(__file__).resolve().parent.parent)
data_path = base_dir+"/base_datasets/questions_about_the_world"
save_base_path = base_dir+"/evaluation_scripts/results/qaw_sft_fixed_2"
os.makedirs(save_base_path, exist_ok=True)
data_wrapper = DatasetWrapper(data_path)
data_wrapper.shuffle(42)

samples = 1280
ds = data_wrapper.get_eval_split(samples=samples)

response_list = []
for sample in ds:
    response_list.append(sample["first_turn"].replace(sample["question"]+"\n", ""))

dir_path = "/evaluation_scripts/results/old/qaw_sft_fixed"
data_list = load_results(dir_path)

for data in tqdm(data_list):
    df = data[1]
    df = df.drop('sem_score', axis=1)
    df = df.drop('sem_scores', axis=1)
    df = df.drop('bert_score', axis=1)
    df = df.drop('bert_scores', axis=1)
    df = df.drop('grammar_err_diff', axis=1)
    messages = df["messages"]
    pred_grammar_errors = df["pred_grammar_errors"]
    ref_grammar_errors = df["ref_grammar_errors"]
    sem_score_list = []
    ref_list = []
    pred_list = []
    test = (data[0], df)
    for i in tqdm(range(samples)):
        ref = response_list[i]
        pred = split_messages(messages[i])[-1]
        recall, precision, f1 = get_sem_score(pred, ref)
        sem_score = { "recall" : recall, "precision" : precision, "f1" : f1}
        sem_score_list.append(sem_score)
        ref_list.append(ref)
        pred_list.append(pred)
        test_1 = pred_grammar_errors[i]
        test_2 = ref_grammar_errors[i]
        test_3 = len(test_1)
        test_4 = len(test_2)
        test = test_3 - test_4
        pass
    bert_score_dict = get_bert_score(pred_list, ref_list)
    bert_score_list = []
    for i in range(samples):
        bert_score_list.append({ "recall" : bert_score_dict["recall"][i], "precision" : bert_score_dict["precision"][i], "f1" : bert_score_dict["f1"][i]})
    df.insert(0, "bert_score_gpt_response", bert_score_list)
    df.insert(0, "sem_score_gpt_response", sem_score_list)
    with open(save_base_path+"/"+data[0], mode="wb") as f:
        pickle.dump(df, file=f)


