import random
import torch
import numpy as np
import os
import copy
from accelerate.utils import set_seed

from config_hub.dataset_config_hub.config import USER_TOKEN, SYSTEM_TOKEN, CONTEXT_TOKEN
from config_hub.dataset_config_hub.config import USER_SITUATION_TOKEN, USER_EMOTION_TOKEN, USER_ACTION_TOKEN, USER_KNOWLEDGE_TOKEN
from lit_llama.tokenizer import Tokenizer


def set_random_seed(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 环境参数的哈希值
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # 单gpu
    torch.cuda.manual_seed_all(seed) # 多gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # 使用非确定性算法


def unique_ordered_list(lst: list):
    all_items = list(set(lst))
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    assert len(result) == len(all_items), "The list is illenegal"
    return result


def generate_prompt(dialogue_context: str):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return (
        "Below is a target-driven conversation between the user and AI. "
        "Extract the intention of user's latest utterance.\n\n"
        f"### Conversation:\n{dialogue_context}\n\n### Intention: "
    )


def convert_example_to_feature_for_consint(instance: dict,
                                           tokenizer: Tokenizer, 
                                           max_seq_length: int = 1024,
                                           mask_inputs: bool = True,
                                           is_test: bool = False):
    """
    function that convert an instance to input and labels for a response generation model.
    @param tokenizer: a huggingface tokenizer
    @param instance: an instance from the data.
    @param max_sequence_length: the maximum length of the input sequence.
    @param max_target_length: the maximum length of the target response
    @param is_test: True if inference or False if training.
    @return: an input sequence and its corresponding labels.
    """

    dialogue_context = instance['dialogue_context']
    dialogue_str = ""
    prev_convints = instance['pre_gpt-3.5-intention'][1:]

    idx = 0
    num_prev_convints = len(prev_convints)

    for utt in dialogue_context:
        
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN + " "
            dialogue_str += utt['content'] + " "

            if idx < num_prev_convints:
                dialogue_str += USER_SITUATION_TOKEN + " " + prev_convints[idx][USER_SITUATION_TOKEN] + " "
                dialogue_str += USER_EMOTION_TOKEN + " " + prev_convints[idx][USER_EMOTION_TOKEN] + " "
                dialogue_str += USER_ACTION_TOKEN + " " + prev_convints[idx][USER_ACTION_TOKEN] + " "
                dialogue_str += USER_KNOWLEDGE_TOKEN + " " + prev_convints[idx][USER_KNOWLEDGE_TOKEN] + " "
                idx += 1
        
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN + " "
            dialogue_str += utt['content'] + " "

    target = instance['task_background']['target_topic']
    # construct the input sequence for response generation task
    # for durecdial dataset
    # if dataset == 'durecdial':
    #     input_str = f"{CONTEXT_TOKEN}: {dialogue_str}"
    # # for inspired dataset
    # elif dataset == 'inspired':
    #     input_str = f"{CONTEXT_TOKEN}: {dialogue_str}"

    input_str = f"{CONTEXT_TOKEN}: {dialogue_str}"

    # input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    # input_ids = input_ids[-(max_sequence_length - 2):]
    # input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    full_prompt_str = generate_prompt(input_str)

    # construct the label for response generation task
    if not is_test:
        label_str = USER_SITUATION_TOKEN + " " + instance['gpt-3.5-intention'][USER_SITUATION_TOKEN] + " "
        label_str += USER_EMOTION_TOKEN + " " + instance['gpt-3.5-intention'][USER_EMOTION_TOKEN] + " "
        label_str += USER_ACTION_TOKEN + " " + instance['gpt-3.5-intention'][USER_ACTION_TOKEN] + " "
        label_str += USER_KNOWLEDGE_TOKEN + " " + instance['gpt-3.5-intention'][USER_KNOWLEDGE_TOKEN] 
    else:
        label_str = USER_SITUATION_TOKEN + " " + instance['gpt-4-intention'][USER_SITUATION_TOKEN] + " "
        label_str += USER_EMOTION_TOKEN + " " + instance['gpt-4-intention'][USER_EMOTION_TOKEN] + " "
        label_str += USER_ACTION_TOKEN + " " + instance['gpt-4-intention'][USER_ACTION_TOKEN] + " "
        label_str += USER_KNOWLEDGE_TOKEN + " " + instance['gpt-4-intention'][USER_KNOWLEDGE_TOKEN] 
    
    # label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label_str))        
    # label = label[:max_target_length]
    # label = label + [tokenizer.eos_token_id]

    full_prompt_with_convint_str = full_prompt_str + label_str

    # the pad is set to false, so it will not expand to the max_seq_length
    encoded_full_prompt = tokenizer.encode(string=full_prompt_str,
                                           bos=True,
                                           eos=False)
    # the pad is set to true, so it will expand to the max_seq_length
    encoded_full_prompt_with_convint = tokenizer.encode(string=full_prompt_with_convint_str,
                                                        bos=True,
                                                        eos=True,
                                                        max_length=max_seq_length)
    
    encoded_label_features = encoded_full_prompt_with_convint.clone()
    
    if mask_inputs:
        encoded_label_features[: len(encoded_full_prompt)-1] = tokenizer.pad_id

    # if len(dialogue_context) <= 9:
        
    #     # print("=====================================================")
    #     # print(encoded_full_prompt_with_convint)
    #     # print(tokenizer.decode(encoded_full_prompt_with_convint))

    #     # print("=====================================================")
    #     # print(encoded_full_prompt)
    #     # print(tokenizer.decode(encoded_full_prompt))

    #     # print("=====================================================")
    #     # print(encoded_label_features)
    #     # print(tokenizer.decode(torch.where(encoded_label_features == tokenizer.pad_id, torch.tensor(0), encoded_label_features)))

    #     # print("=====================================================")
    #     # print(tokenizer.decode(torch.tensor([29871])))
    #     # print(tokenizer.decode(torch.tensor([29901, 29871])))
    #     # print("=====================================================")
    #     # print(tokenizer.decode(torch.tensor([29901, 29871]))[0])
    #     # print(tokenizer.decode(torch.tensor([29901, 29871]))[1])
    #     # print("prev_convints: ")
    #     # print(instance['pre_gpt-3.5-intention'])
    #     # print("=====================================================")
    #     # print("full_prompt_str: ")

    #     # print(full_prompt_with_convint_str)
    #     # print("=====================================================")
        

        
    # else:
    #     exit()

    features = {
        "full_prompt_str": full_prompt_str,
        "label_str": label_str,
        "input_ids": encoded_full_prompt_with_convint,
        "input_ids_no_convint": encoded_full_prompt,
        "label": encoded_label_features
    }

    return features