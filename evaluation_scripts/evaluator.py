from sympy.solvers.diophantine.diophantine import length

from utils.get_sem_score import get_sem_score
from utils.eval_utils import get_length, format_question, dynamic_length_generation
from utils.language_tool_utils import lan_tool_grammar_check
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import pickle
from transformers.pipelines.pt_utils import KeyDataset
from threading import Thread
from utils.bertscore import get_bert_score
now = datetime.now()



class ModelEvaluator:

    def __init__(self, pipeline, eval_modes, language, save_loc, samples, batch_size, ds, tokenizer_for_chat_template,
                 best_of_n_mode=False, custom_llama_mode=False, use_eval_question_strs=False, use_dynamic_length=False,
                 dyn_len_params=None, fixed_prompt_mode=False, fixed_len=None, custom_sys_prompt=None):
        self.pipe = pipeline
        self.eval_modes = eval_modes
        self.current_eval_mode_index = 0
        self.language = language
        self.save_loc = save_loc
        self.samples = samples
        self.batch_size = batch_size
        self.best_of_n_mode = best_of_n_mode # pick best response from set of n responses
        self.custom_llama_mode = custom_llama_mode # adjust output token probability distribution to favour eos-token when close to target (see also utils.custom_llama_3_1.py)
        self.fixed_prompt_mode = fixed_prompt_mode # set to true if prompt is already in correct format to be used as input
        self.fixed_len = fixed_len # set if target lenth of all samples should be the same value, if None, target length will instead be generated from length of target text
        self.custom_sys_prompt = custom_sys_prompt # add additional stuff to the system prompt
        self.use_eval_question_strs = use_eval_question_strs # use alternative strings to specify length targets; only relevant if prompts aren't fixed
        self.use_dynamic_length = use_dynamic_length #use dynamic length targets
        self.dyn_len_params = dyn_len_params #params for dynamic length targets, will be passed as is to dynamic_length_generation() (see utils.eval_utils.py)
        self.result_df = pd.DataFrame(
            {"messages": [], "bert_score_gpt_response": [], "sem_score_gpt_response": [], "len_actual": [], "len_target":[],  "len_diff": [], "pred_grammar_errors": [],
             "ref_grammar_errors": [], "mode": [], "base_question" : []})
        self.tokenizer_for_chat_template = tokenizer_for_chat_template #this should be the base Llama 3.1 tokenizer and is used to avoid bugs with the unsloth tokenizer
        self.ds=ds.map(self.apply_template)



    def get_next_eval_mode(self):
        i = self.current_eval_mode_index
        if(i == len(self.eval_modes)-1):
            self.current_eval_mode_index = 0
        else:
            self.current_eval_mode_index += 1
        return self.eval_modes[i]


    def apply_template(self, data):
        if self.custom_sys_prompt is not None:
            formatted_input = [{"role":"system", "content": self.custom_sys_prompt}]
        else:
            formatted_input = []

        if self.fixed_prompt_mode:
            input_list = data["input"]

            for i, str in enumerate(input_list):
                if i%2==1:
                    formatted_input.append({"role": "user", "content": str})
                else:
                    formatted_input.append({"role": "assistant", "content": str})
            input_str = self.tokenizer_for_chat_template.apply_chat_template(formatted_input, tokenize=False, add_generation_prompt=True)
            base_question = input_str
            length = self.fixed_len
            ref = data["target_output"]
            eval_mode = self.get_next_eval_mode()
        else:
            query = data["question"]
            ref = data["first_turn"].replace(query + "\n", "")
            eval_mode = self.get_next_eval_mode()
            length = get_length(ref, eval_mode, self.language)
            query_str = format_question(query, eval_mode, length, eval_questions=self.use_eval_question_strs)
            formatted_input.append({"role": "user", "content": query_str})
            input_str = self.tokenizer_for_chat_template.apply_chat_template(formatted_input, tokenize=False, add_generation_prompt=True)
            base_question = self.tokenizer_for_chat_template.apply_chat_template([{"role": "user", "content": data["question"]}],tokenize=False, add_generation_prompt=True)
        return {"input" : input_str, "base_question": base_question, "length" : length, "ref" : ref, "eval_mode" : eval_mode}

    def get_output(self, batch):
        #generates model outputs for inputs in batch, with behaviour depending on parameters passed when initializing the evaluator
        output_strs = []
        preds = []
        if self.use_dynamic_length:
            curr_target_lens = []
            for i, data in enumerate(batch):
                target_len = data["length"]
                eval_mode = data["eval_mode"]
                prompt = data["question"]
                pred, output_str, curr_target_len = dynamic_length_generation(self.pipe, self.tokenizer_for_chat_template, prompt, target_len, eval_mode, self.language,
                                                                              max_tokens_between_target_change=self.dyn_len_params["max_tokens_between_target_change"],
                                                                              min_tokens_between_target_change=self.dyn_len_params["min_tokens_between_target_change"],
                                                                              min_remaining_length=self.dyn_len_params["min_remaining_length"],
                                                                              length_change_scale=self.dyn_len_params["length_change_scale"])
                output_strs.append(output_str)
                preds.append(pred)
                curr_target_lens.append(curr_target_len)
            return output_strs, preds, curr_target_lens
        elif self.best_of_n_mode and not self.custom_llama_mode:
            output_strs, preds = self.pipe.batched_generate(batch, self.batch_size, self.language)
        elif self.best_of_n_mode and self.custom_llama_mode:
            for i, data in enumerate(batch):
                input_str = data["input"]
                target_len = data["length"]
                eval_mode = data["eval_mode"]
                output_str, pred = self.pipe.generate(input_str, target_len, eval_mode, self.language)
                output_strs.append(output_str)
                preds.append(pred)

        elif self.custom_llama_mode:
            for i, data in enumerate(batch):
                input_str = data["input"]
                target_len = data["length"]
                eval_mode = data["eval_mode"]
                pred = self.pipe.generate_from_prompt(input_str, target_len, eval_mode, self.language)
                output_str = input_str + "\n" + pred
                output_strs.append(output_str)
                preds.append(pred)
        else:
            for i, output in enumerate(self.pipe(KeyDataset(batch, "input"), batch_size=self.batch_size, truncation="only_first")):
                output_str = output[0]["generated_text"]
                pred = output_str.replace(batch["input"][i],"")
                output_strs.append(output_str)
                preds.append(pred)
        return output_strs, preds

    def data_collection(self, batch, preds, results):
        # collects various data points based on the information in batch and the corresponding model predictions
        pred_grammar_errors_list = []
        ref_grammar_errors_list = []
        for i in range(len(preds)):
            pred_grammar_errors = lan_tool_grammar_check(preds[i])
            ref_grammar_errors = lan_tool_grammar_check(batch["ref"][i])
            pred_grammar_errors_list.append(pred_grammar_errors)
            ref_grammar_errors_list.append(ref_grammar_errors)
        results["pred_grammar_errors"] = pred_grammar_errors_list
        results["ref_grammar_errors"] = ref_grammar_errors_list
        results["len_actual"] = []
        results["len_target"] = []
        results["len_diff"] = []
        results["mode"] = []
        results["base_question"] = []
        results["sem_score_gpt_response"] = []
        results["bert_score_gpt_response"] = []
        for i in range(len(batch)):
            eval_mode = batch[i]["eval_mode"]
            len_actual = get_length(preds[i], eval_mode, self.language)
            len_target = batch[i]["length"]
            len_diff = abs(len_target-len_actual)
            results["len_actual"].append(len_actual)
            results["len_target"].append(len_target)
            results["len_diff"].append(len_diff)
            results["mode"].append(eval_mode)
            results["base_question"].append(batch[i]["base_question"])
            recall, precision, f1 = get_sem_score(preds[i], batch["ref"][i])
            results["sem_score_gpt_response"].append({ "recall" : recall, "precision" : precision, "f1" : f1})
        results["bert_score_gpt_response"] = []
        bert_scores = get_bert_score(preds, batch["ref"])
        for i in range(len(bert_scores["precision"])):
            results["bert_score_gpt_response"].append({"precision": bert_scores["precision"][i], "recall": bert_scores["recall"][i], "f1": bert_scores["f1"][i]})

    def eval_model(self, save_data = True):
        # this is threaded to make it faster
        # one thread handles generating responses, the other handles evaluating them
        batch = Dataset.from_dict(self.ds[0:self.batch_size])
        if self.use_dynamic_length:
            output_strs, preds, curr_target_lens = self.get_output(batch)
        else:
            output_strs, preds = self.get_output(batch)

        for idx in tqdm(range(self.batch_size, self.samples, self.batch_size)):
            if self.use_dynamic_length:
                results = {"messages": output_strs, "len_target_end": curr_target_lens}
            else:
                results = {"messages": output_strs}

            data_collection_thread = Thread(target=self.data_collection, args=(batch, preds, results))
            data_collection_thread.start()

            batch = Dataset.from_dict(self.ds[idx: idx + self.batch_size])
            if self.use_dynamic_length:
                output_strs, preds, curr_target_lens = self.get_output(batch)
            else:
                output_strs, preds = self.get_output(batch)

            data_collection_thread.join()
            self.result_df = pd.concat([self.result_df, pd.DataFrame(results)], ignore_index=True)

        if self.use_dynamic_length:
            results = {"messages": output_strs, "len_target_end": curr_target_lens}
        else:
            results = {"messages": output_strs}
        self.data_collection(batch, preds, results)
        self.result_df = pd.concat([self.result_df, pd.DataFrame(results)], ignore_index=True)
        if save_data:
            self.save_results()
        return self.result_df.copy(deep=True)

    def save_results(self):
        with open(self.save_loc, mode="wb") as f:
            pickle.dump(self.result_df, file=f)