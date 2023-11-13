import os
import json
from loguru import logger
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from transformers import XLMRobertaTokenizerFast, default_data_collator
from transformers import (
    HfArgumentParser,
    set_seed,
)
from arguments import ModelArguments, DataArguments
from modeling import MultiTaskMRCModel, MultiTaskMRCOutput
from data_mrc import get_dataset as get_mrc_dataset
from split_sentence import SplitSentence



def compute_recall(predict_spans:List[Tuple[int]], answer_spans:List[Tuple[int]]):
    """
    计算recall, 计算方式是计算predict_span和answer_span的交集的长度除以answer_span的长度
    """
    total_recall = 0
    for predict_span in predict_spans:
        for answer_span in answer_spans:
            if predict_span[0] < answer_span[1] and predict_span[1] > answer_span[0]:
                total_recall += min(predict_span[1], answer_span[1]) - max(predict_span[0], answer_span[0])
    total_recall /= sum([span[1] - span[0] for span in answer_spans])
    return total_recall


def evaluate(eval_data_file:str):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(path, 'predict_args.json')
    model_args, data_args, training_args = parser.parse_json_file(json_file=json_file)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    # sample_mapping: [n,], offset_mapping: [n, seq_len, 2], sequence_ids_mapping: [n, seq_len], origin_dataset: [n,]
    eval_dataset, sample_mapping, offset_mapping, sequence_ids_mapping, ori_dataset = get_mrc_dataset(eval_data_file, tokenizer)
    logger.info("sample_mapping: {}".format(sample_mapping))
    _model_class = MultiTaskMRCModel
    model = _model_class.from_pretrained(model_args, data_args, training_args,
                                      model_args.model_name_or_path,
                                      cache_dir=model_args.cache_dir,
                                      task_list=['mrc'])
    model.to(device)
    data_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=default_data_collator)
    model.eval()
    all_labels = [] # [batch, seq_len]
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            result:MultiTaskMRCOutput = model(batch)
            logits = result.mrc_logits.cpu().numpy()    # batch, seq_len, 2
            labels = logits.argmax(axis=-1)
            all_labels.extend(labels)
    
    # 获取context中每个word的label
    context_word_labels = [[0 for _ in context] for context in ori_dataset['context']]  # batch, seq_len
    for i, sample_idx in enumerate(sample_mapping):
        sequence_ids = sequence_ids_mapping[i]
        for j, (sequence_id, (token_start, token_end)) in enumerate(zip(sequence_ids, offset_mapping[i])):
            if sequence_id != 1:
                continue
            for k in range(token_start, token_end):
                try:
                    context_word_labels[sample_idx][k] = all_labels[i][j]
                except:
                    logger.info("here")
    # 将context进行分句，如果每个句子中label为1的word的数量超过一半，则该句子作为答案，然后将答案的span进行保存
    predict_spans_list = []   # 每个样本的answer_span
    split_sentence = SplitSentence()
    sentences_list = []
    for sample_idx, context in enumerate(ori_dataset['context']):
        sentences = split_sentence(context, criterion='coarse', max_sen_len=128, min_sen_len=16)
        sentences_list.append(sentences)
        total_len = sum([len(sen) for sen in sentences])
        assert total_len == len(context)
        sentence_word_label_cnt = {i:0 for i in range(len(sentences))}
        sentence_span = {}  # 每个句子的span
        cur_idx = 0
        for i, sen in enumerate(sentences):
            span_start = cur_idx
            span_end = cur_idx + len(sen)
            sentence_span[i] = (span_start, span_end)
            for j in range(len(sen)):
                if context_word_labels[sample_idx][cur_idx] == 1:
                    sentence_word_label_cnt[i] += 1
                cur_idx += 1
        predict_spans_for_one_sample = []
        # 如果一个句子中超过一般的word的label为1，则该句子作为答案
        for i, sen in enumerate(sentences):
            if sentence_word_label_cnt[i] * 2 >= len(sen):
                predict_spans_for_one_sample.append(sentence_span[i])
        # 合并相邻的span
        spans_new = []
        for i, span in enumerate(predict_spans_for_one_sample):
            if i == 0:
                spans_new.append(span)
            else:
                if span[0] == spans_new[-1][1]:
                    spans_new[-1] = (spans_new[-1][0], span[1])
                else:
                    spans_new.append(span)
        predict_spans_list.append(spans_new)

    # 计算recall
    answer_spans_list = ori_dataset['answer_span']
    total_recall = 0
    for predict_spans, answer_spans in zip(predict_spans_list, answer_spans_list):
        total_recall += compute_recall(predict_spans, answer_spans)
    total_recall /= len(answer_spans_list)
    print("recall: ", total_recall)

def main():
    path = os.path.dirname(os.path.abspath(__file__))
    eval_data_file = os.path.join(path, 'data', 'baidu_search_small_standard.jsonl')
    evaluate(eval_data_file)


if __name__ == '__main__':
    main()