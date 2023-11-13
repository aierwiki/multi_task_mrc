import logging
import os
from pathlib import Path
import light_hf_proxy

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


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file='./args.json')
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    _model_class = MultiTaskMRCModel
    model = _model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        train_group_size = data_args.train_group_size,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        task_list=['mrc'],
        freeze_hf_model=True
    )
    training_args.remove_unused_columns=False
    train_dataset, sample_mapping, offset_mapping, sequence_ids_mapping, ori_dataset = get_mrc_dataset(data_args.train_data, tokenizer)
    _trainer_class = MultiTaskMRCTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
