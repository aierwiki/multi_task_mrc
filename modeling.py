import logging
from typing import List
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


class MultiTaskMRCModel(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, train_task_list:List=['mrc']):
        """
        train_task_list: list of task names, e.g. ['mrc', 'reranker']，用于指定训练的任务，如果为None的话，表示训练所有任务
        """
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )
        # 下面定义一层transformer编码器，使用SequenceClassifierOutput的hidden_states[-1]作为输入，做序列标注任务。d_model和nhead使用与hf_model一样的配置
        d_model = self.hf_model.config.hidden_size
        nhead = self.hf_model.config.num_attention_heads
        self.sequence_labeling_head = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)


    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
