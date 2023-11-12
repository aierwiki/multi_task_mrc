import os
import json
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, DataCollatorForTokenClassification
from transformers import XLMRobertaTokenizerFast, default_data_collator
from transformers import (
    HfArgumentParser,
    set_seed,
)
from arguments import ModelArguments, DataArguments
from modeling import MultiTaskMRCModel
from trainer import MultiTaskMRCTrainer
from data_mrc import get_dataset as get_mrc_dataset



def evaluate(eval_data_file:str):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    json_file = os.path.join(path, 'predict_args.json')
    model_args, data_args, training_args = parser.parse_json_file(json_file=json_file)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    eval_dataset, sample_mapping, offset_mapping, ori_dataset = get_mrc_dataset(json_file, tokenizer)
    _model_class = MultiTaskMRCModel
    model = _model_class.from_pretrained(model_args, data_args, training_args,
                                      model_args.model_name_or_path,
                                      cache_dir=model_args.cache_dir,
                                      task_list=['mrc'])
    model.to(device)
    data_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=default_data_collator)
    model.eval()
    all_labels = []
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            result:MultiTaskMRCOutput = model(batch)
            logits = result.mrc_logits.cpu().numpy()    # batch, seq_len, 2
            labels = logits.argmax(axis=-1)
            all_labels.extend(labels)
    
    # 获取context中每个word的label
    context_word_labels = [[0 for _ in context] for context in ori_dataset['context']]
    for i, sample_idx in enumerate(sample_mapping):
        sequence_ids = tokenized_examples.sequence_ids(i)
        for j, (sequence_id, (token_start, token_end)) in enumerate(zip(sequence_ids, offset_mapping[i])):
            if sequence_id != 1:
                continue
            for k in range(token_start, token_end):
                context_word_labels[sample_idx][k] = labels[i][j]
    # 将context进行分句，如果每个句子中label为1的word的数量超过一半，则该句子作为答案，然后将答案的span进行保存
    answer_spans_list = []   # 每个样本的answer_span
    split_sentence = SplitSentence()
    sentences_list = []
    for context in ori_dataset['context']:
        sentences_list.append(split_sentence(context, criterion='coarse', max_sen_len=128, min_sen_len=16))
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
        answer_spans_list.append(answer_spans_new)

    # 计算f1
    total_f1 = 0
    for answer_spans, ori_answer_spans in zip(answer_spans_list, ori_dataset['answer_span']):
        total_f1 += compute_f1(answer_spans, ori_answer_spans)
    total_f1 /= len(answer_spans_list)
    print("f1: ", total_f1)

def main():
    evaluate()


if __name__ == '__main__':
    main()