import os
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput, ModelOutput

from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskMRCOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reranker_logits: Optional[torch.FloatTensor] = None   # 用于reranker任务的logits
    mrc_logits: Optional[torch.FloatTensor] = None        # 用于mrc任务的logits, 每个位置的token属于答案的logits


class MRCHead(nn.Module):
    """由一层transformer编码器和一层线性层组成，用于做序列标注任务"""
    def __init__(self, d_model, nhead, num_labels):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states, labels=None):
        hidden_states = self.transformer_layer(hidden_states)
        logits = self.linear(hidden_states)
        if self.training:
            loss = nn.CrossEntropyLoss(reduction='mean')(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits


class MultiTaskMRCModel(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, mrc_head: MRCHead, train_group_size = None, per_device_train_batch_size = 1, 
                 task_list:List=['mrc'], freeze_hf_model=False):
        """
        task_list: list of task names, e.g. ['mrc', 'reranker']，用于指定训练的任务，如果为None的话，表示训练所有任务
        """
        super().__init__()
        self.hf_model = hf_model
        self.train_group_size = train_group_size
        self.per_device_train_batch_size = per_device_train_batch_size
        self.task_list = task_list
        self.freeze_hf_model = freeze_hf_model
        if self.freeze_hf_model:
            for param in self.hf_model.parameters():
                param.requires_grad = False
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.per_device_train_batch_size, dtype=torch.long)
        )
        self.mrc_head = mrc_head
       

    def forward(self, batch):
        mrc_labels = batch.pop('mrc_labels', None)
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True, output_hidden_states=True)
        reranker_logits = ranker_out.logits
        hidden_states = ranker_out.hidden_states
        output = MultiTaskMRCOutput()
        if self.training:
            total_loss = None
            if 'reranker' in self.task_list or self.task_list is None:
                scores = reranker_logits.view(
                    self.per_device_train_batch_size,
                    self.train_group_size
                )
                reranker_loss = self.cross_entropy(scores, self.target_label)
                if total_loss is None:
                    total_loss = reranker_loss
                else:
                    total_loss += reranker_loss
                output.reranker_logits = reranker_logits
            if 'mrc' in self.task_list or self.task_list is None and mrc_labels is not None:
                # 将hidden_states[-1]作为输入，做序列标注任务
                mrc_loss, mrc_logits = self.mrc_head(hidden_states[-1], mrc_labels)
                output.mrc_logits = mrc_logits
                if total_loss is None:
                    total_loss = mrc_loss
                else:
                    total_loss += mrc_loss
            output.loss = total_loss
        else:
            if 'reranker' in self.task_list or self.task_list is None:
                output.reranker_logits = reranker_logits
            if 'mrc' in self.task_list or self.task_list is None:
                # 将hidden_states[-1]作为输入，做序列标注任务
                mrc_logits = self.mrc_head(hidden_states[-1])
                output.mrc_logits = mrc_logits
        return output

    @classmethod
    def from_pretrained(
            cls, *args, **kwargs
    ):
        train_group_size = kwargs.pop('train_group_size', 4)
        per_device_train_batch_size = kwargs.pop('per_device_train_batch_size', 1)
        task_list = kwargs.pop('task_list', ['mrc'])
        freeze_hf_model = kwargs.pop('freeze_hf_model', False)
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        if 'mrc' in task_list or task_list is None:
            # 加载mrc_head
            mrc_head = MRCHead(d_model=hf_model.config.hidden_size, nhead=hf_model.config.num_attention_heads, num_labels=3)
            model_name_or_path = kwargs.get('model_name_or_path', None)
            if model_name_or_path is not None:
                mrc_head.load_state_dict(torch.load(os.path.join(model_name_or_path, 'mrc_head.bin')))
        else:
            mrc_head = None
        multi_task_model = cls(hf_model, mrc_head, train_group_size=train_group_size, per_device_train_batch_size=per_device_train_batch_size, 
                               task_list=task_list, freeze_hf_model=freeze_hf_model)
        return multi_task_model

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        # 保存mrc_head
        torch.save(self.mrc_head.state_dict(), os.path.join(output_dir, 'mrc_head.bin'))
