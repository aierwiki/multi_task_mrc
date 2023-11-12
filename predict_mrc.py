import os
import json
from loguru import logger
from transformers import XLMRobertaTokenizerFast, AutoModel
from transformers import TrainingArguments, HfArgumentParser
from arguments import ModelArguments, DataArguments
from modeling import MultiTaskMRCModel, MultiTaskMRCOutput
from split_sentence import SplitSentence


def predict(query, context, max_length=512, stride=32):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file='./predict_args.json')
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True
    )
    _model_class = MultiTaskMRCModel
    model = _model_class.from_pretrained(model_args, data_args, training_args,
                                      model_args.model_name_or_path,
                                      cache_dir=model_args.cache_dir,
                                      task_list=['mrc'])
    tokenized_examples = tokenizer(query,
                                   context,
                                   truncation="only_second",
                                   max_length=max_length,
                                   stride=stride,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding="max_length")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")   # 类似于[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    offset_mapping = tokenized_examples.pop("offset_mapping")   # 类似于[[(0, 0), (1, 2), (3, 6)], ...]   每个token在对应context文本中的位置

    result:MultiTaskMRCOutput = model(tokenized_examples)
    logits = result.mrc_logits.cup().numpy()    # batch, seq_len, 2
    labels = logits.argmax(axis=-1)
    # 获取context中每个word的label
    context_word_labels = [0 for _ in context]
    for i, sample_idx in enumerate(sample_mapping):
        sequence_ids = tokenized_examples.sequence_ids(i)
        for j, (sequence_id, (token_start, token_end)) in enumerate(zip(sequence_ids, offset_mapping[i])):
            if sequence_id != 1:
                continue
            context_word_labels[token_start:token_end] = labels[i][j]
    # 将context进行分句，如果每个句子中label为1的word的数量超过一半，则该句子作为答案，然后将答案的span进行保存
    answer_spans = []
    split_sentence = SplitSentence()
    sentence_list = split_sentence.split_sentence(context, criterion='coarse', max_sen_len=128, min_sen_len=16)
    total_len = sum([len(sen) for sen in sentence_list])
    assert total_len == len(context)
    sentence_word_cnt = {i:0 for i in range(len(sentence_list))}
    sentence_span = {}
    cur_idx = 0
    for i, sen in enumerate(sentence_list):
        span_start = cur_idx
        span_end = cur_idx + len(sen)
        sentence_span[i] = (span_start, span_end)
        for j in range(len(sen)):
            if context_word_labels[cur_idx] == 1:
                sentence_word_cnt[i] += 1
            cur_idx += 1
    for i, sen in enumerate(sentence_list):
        if sentence_word_cnt[i] * 2 >= len(sen):
            answer_spans.append(sentence_span[i])
    # 合并相邻的span
    answer_spans_new = []
    for i, span in enumerate(answer_spans):
        if i == 0:
            answer_spans_new.append(span)
        else:
            if span[0] == answer_spans_new[-1][1]:
                answer_spans_new[-1] = (answer_spans_new[-1][0], span[1])
            else:
                answer_spans_new.append(span)
    return answer_spans_new


def main():
    query = '中国的首都是哪里？'
    context = '北京是中国的首都，上海是中国的经济中心，深圳是中国的创新中心。'
    answer_spans = predict(query, context)
    print("答：")
    for span in answer_spans:
        print(context[span[0]:span[1]])


if __name__ == "__main__":
    main()