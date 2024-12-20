import torch.nn.functional as F
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from torch.distributions.categorical import Categorical
from accelerate import PartialState
from utils.eval_utils import get_length

device = PartialState().process_index




class CustomLlama31:
    '''
    customized llama that can adjust the probability of the eos-token (=token that ends generation) in its output
    distribution based on the difference between the current length of it's output and a target length.
    '''
    def __init__(self, model = None, tokenizer = None):
        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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", device_map="auto")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="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", device_map="auto")


    def _generate_next_token(self, input, eos_token_factor, temp=1.):
        #fetch output token distribution
        out = self.model.generate(input, max_new_tokens=1, output_scores=True, return_dict_in_generate=True, pad_token_id=self.tokenizer.eos_token_id, temperature=temp)
        logits = out.scores[0][0].detach().clone()

        #adjust eos-token probability based on eos_token_factor when it isn't -inf
        if not torch.isinf(logits[self.tokenizer.eos_token_id]):
            logits[self.tokenizer.eos_token_id] = logits[self.tokenizer.eos_token_id]*eos_token_factor

        #sample new distribution
        probs = F.softmax(logits, dim=0)
        dist = Categorical(probs)
        chosen_token = dist.sample()

        return chosen_token.cpu().item()

    def _get_len_ratio(self, tokens, target_len, eval_mode, language):
        #returns ratio between current output length and target length
        str = self.tokenizer.decode(tokens)
        length = get_length(str, eval_mode=eval_mode, language=language)
        return length/target_len

    def generate_from_prompt(self, prompt_str, target_len, eval_mode, language, temp = 1.):
        input = self.tokenizer(prompt_str)
        input = input.data["input_ids"]
        output = []
        last_token = None
        while last_token != self.tokenizer.eos_token_id:
            input_tensor = torch.tensor(input, dtype=torch.int32).unsqueeze(0).to(device)

            # this is the factor that adjusts the probability of the chosen token being the eos-token
            # it is arbitrarily computed as the ratio between the current and target length to the third power
            # this way it is small before the target length is reached and rises rapidly once it's surpassed
            eos_token_factor = self._get_len_ratio(output, target_len, eval_mode, language)**3

            next_token = self._generate_next_token(input_tensor, eos_token_factor, temp=temp)
            input.append(next_token)
            output.append(next_token)
            last_token = next_token
        return self.tokenizer.decode(output)

