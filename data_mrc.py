import math
import os
import random
from typing import List, Tuple, Dict
import numpy as np
import datasets
from transformers import XLMRobertaTokenizerFast, AutoTokenizer

from arguments import DataArguments


def prepare_features(examples: List[dict], tokenizer: XLMRobertaTokenizerFast, max_length=512, stride=32):
    """
    {
        "query": "渝北区面积", 
        "context": "渝北区，重庆市辖区，属重庆主城区、重庆大都市区，地处重庆市西北部。东邻长寿区、南与江北区毗邻，同巴南区、南岸区、沙坪坝区隔江相望，西连北碚区、合川区，北接四川省广安市华蓥市、邻水县，总面积1452.03平方千米。"
        "answer_span": [[10, 20], [40, 50]]
    }
    """
    tokenized_examples = tokenizer(examples["query"], 
                                   examples["context"], 
                                   truncation="only_second", 
                                   max_length=max_length,
                                   stride=stride,
                                   return_overflowing_tokens=True, 
                                   return_offsets_mapping=True, 
                                   padding="max_length")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")   # 类似于[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    offset_mapping = tokenized_examples.pop("offset_mapping")   # 类似于[[(0, 0), (1, 2), (3, 6)], ...]   每个token在对应context文本中的位置
    # 根据answer_span构造label
    answer_span_list = examples["answer_span"]
    labels = []
    for i, (sample_idx, one_example_offset_mapping) in enumerate(zip(sample_mapping, offset_mapping)):
        one_sample_labels = []
        sequence_ids = tokenized_examples.sequence_ids(i)
        for j, (sequence_id, (token_start, token_end)) in enumerate(zip(sequence_ids, one_example_offset_mapping)):
            if sequence_id != 1:
                one_sample_labels.append(-100)
                continue

            if token_start == 0 and token_end == 0: # special tokens 直接过滤
                one_sample_labels.append(-100)
                continue

            spans = answer_span_list[sample_idx]
            # 如果[token_start,token_end)区间和spans中的任意一个span有交集，则label为1，否则为0
            label = 0
            for span in spans:
                if token_start < span[1] and token_end > span[0]:
                    label = 1
                    break
            one_sample_labels.append(label)
        labels.append(one_sample_labels)
    tokenized_examples["mrc_labels"] = labels
    return tokenized_examples

    


def get_dataset(args: DataArguments, tokenizer: XLMRobertaTokenizerFast):
    """
    {
        "query": "渝北区面积", 
        "context": "渝北区，重庆市辖区，属重庆主城区、重庆大都市区，地处重庆市西北部。东邻长寿区、南与江北区毗邻，同巴南区、南岸区、沙坪坝区隔江相望，西连北碚区、合川区，北接四川省广安市华蓥市、邻水县，总面积1452.03平方千米。"
        "answer_span": [[10, 20], [40, 50]]
    }
    """
    if os.path.isdir(args.train_data):
        train_datasets = []
        for file in os.listdir(args.train_data):
            temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                    split='train')
            train_datasets.append(temp_dataset)
        dataset = datasets.concatenate_datasets(train_datasets)
    else:
        dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')
    
    tokenized_dataset = dataset.map(prepare_features, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=dataset.column_names)

    return tokenized_dataset


def main():
    model_name = "BAAI/bge-reranker-base"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    args = DataArguments("./data/baidu_search_small_standard.jsonl")
    dataset = get_dataset(args, tokenizer)
    for i in range(10):
        example = dataset[i]
        input_ids = example["input_ids"]
        input_ids = np.array(input_ids)
        labels = example["mrc_labels"]
        labels = np.array(labels)
        ans_input_ids = input_ids[labels == 1]
        ans = tokenizer.decode(ans_input_ids)
        print(f"ans: {ans}")
        break


if __name__ == "__main__":
    main()