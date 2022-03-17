#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

from functools import partial
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets.arrow_dataset import concatenate_datasets
import numpy as np
from datasets import load_dataset, load_metric
import datasets

import warnings
import torch
warnings.filterwarnings("ignore")

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import T5Tokenizer, T5ForConditionalGeneration

tasks = [
    "cola",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2"
]

def label2text(task_name, label):
    # easy_labels = ("true", "false")
    # return easy_labels[label]
    if task_to_labels[task_name] is None:
        return label
    else:
        return task_to_labels[task_name][label]
        # return easy_labels[label]


def get_optimizer_grouped_parameters(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_EXTENSIONS_DIR'] = os.getcwd()
# os.environ['NCCL_DEBUG'] = "INFO"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", ),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", ),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "axb": ("sentence1", "sentence2"),
    "axg": ("premise", "hypothesis"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
}

logger = logging.getLogger(__name__)



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    print(data_args.dataset_name, data_args.task_name)

    # Labels
    is_regression = False
    num_labels = 2

    # if data_args.task_name is not None:
    #     is_regression = data_args.task_name == "stsb"
    #     if not is_regression:
    #         label_list = raw_datasets["train"].features["label"].names
    #         num_labels = len(label_list)
    #     else:
    #         num_labels = 1
    # else:
    #     # Trying to have good defaults here, don't hesitate to tweak to your needs.
    #     is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    #     if is_regression:
    #         num_labels = 1
    #     else:
    #         # A useful fast method:
    #         # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    #         label_list = raw_datasets["train"].unique("label")
    #         label_list.sort()  # Let's sort it for determinism
    #         num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # print(tokenizer.get_vocab())
    # model = AutoModelForSequenceClassification.from_pretrained(
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # # Some models have set the order of the labels to use, so let's make sure we do use it.
    # label_to_id = None
    # if (
    #     model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    #     and data_args.task_name is not None
    #     and not is_regression
    # ):
    #     # Some have all caps in their config, some don't.
    #     label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
    #         label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #             "\nIgnoring the model labels as a result.",
    #         )
    # elif data_args.task_name is None and not is_regression:
    #     label_to_id = {v: i for i, v in enumerate(label_list)}

    # if label_to_id is not None:
    #     model.config.label2id = label_to_id
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}
    # elif data_args.task_name is not None and not is_regression:
    #     model.config.label2id = {l: i for i, l in enumerate(label_list)}
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples, task_name):
        # Tokenize the texts
        sentence1_key = task_to_keys[task_name][0]
        sentence2_key = None if len(task_to_keys[task_name]) == 1 else task_to_keys[task_name][1]
        sentence1_examples = examples[sentence1_key]
        sentence2_examples = None if sentence2_key is None else examples[sentence2_key]
        processed_examples = []
        for i in range(len(sentence1_examples)):
            elements = [
                    task_name, 
                    sentence1_key+":",
                    sentence1_examples[i],
                ]
            if sentence2_examples is not None:
                elements += [
                    sentence2_key+":",
                    sentence2_examples[i],
                ]
            processed_examples.append(" ".join(elements))

        texts = (
            (processed_examples,)
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True, return_tensors="np")

        if "label" in examples:

            labels = examples["label"]
            labels = [label2text(task_name, label) for label in labels]

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(labels, max_length=2, padding=padding, truncation=True, return_tensors="np")

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            result["labels"] = labels["input_ids"]
        # del result['label']
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = None
        for task_key in task_to_labels.keys():
            dataset = load_dataset(
                data_args.dataset_name, task_key, cache_dir=model_args.cache_dir
            )
            print(dataset)
            dataset = dataset.map(
                partial(preprocess_function, task_name=task_key),
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                remove_columns=["idx", "label"] + list(task_to_keys[task_key])
            )
            if data_args.max_eval_samples is not None and len(dataset['validation']) > data_args.max_eval_samples:
                dataset['validation'] = dataset['validation'].shuffle().select(range(data_args.max_eval_samples))
            if data_args.max_train_samples is not None and len(dataset['train']) > data_args.max_train_samples:
                dataset['train'] = dataset['train'].shuffle().select(range(data_args.max_train_samples))

            if raw_datasets is None:
                raw_datasets = dataset
            else:
                raw_datasets['train'] = concatenate_datasets([raw_datasets['train'], dataset['train']])
                raw_datasets['validation'] = concatenate_datasets([raw_datasets['validation'], dataset['validation']])
                raw_datasets['test'] = concatenate_datasets([raw_datasets['test'], dataset['test']])
        print(raw_datasets)
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"].shuffle()
        print(train_dataset)
        # if data_args.max_train_samples is not None:
        #     train_dataset = train_dataset.shuffle().select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"].shuffle()
        # if data_args.max_eval_samples is not None:
        #     eval_dataset = eval_dataset.shuffle().select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"].shuffle()
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.shuffle().select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    metric = load_metric(data_args.dataset_name, data_args.task_name)

    print("raw_datasets", raw_datasets)
    print("train_dataset", train_dataset)
    print("eval_dataset", eval_dataset)

    # if data_args.task_name == "cola":
    #     training_args.metric_for_best_model = "matthews_correlation"
    # else:
    #     training_args.metric_for_best_model = "accuracy"
    # training_args.greater_is_better = True

    vocab = tokenizer.get_vocab()
    # print(vocab)
    pos_token = tokenizer(task_to_labels[data_args.task_name][1]).input_ids[0]
    neg_token = tokenizer(task_to_labels[data_args.task_name][0]).input_ids[0]

    # print(pos_token, neg_token)
    # exit()
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # print(preds.shape, p.label_ids.shape)
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=-1)

        # for i in range(len(preds)):
        #     pred_sample = preds[i].astype(int)
        #     ref_sample = p.label_ids[i].astype(int)
        #     ref_sample[ref_sample < 0] = 0
        #     print("predict %s, reference %s" % (
        #         tokenizer.decode(pred_sample),
        #         tokenizer.decode(ref_sample),
        #         )
        #     )
        preds = preds[:, 0] == pos_token
        label_ids = p.label_ids[:, 0] == pos_token

        # print(preds, label_ids, pos_token)

        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

        # if data_args.task_name is not None:
        #     result = metric.compute(predictions=preds, references=p.label_ids)
        #     if len(result) > 1:
        #         result["combined_score"] = np.mean(list(result.values())).item()
        #     return result
        # elif is_regression:
        #     return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        # else:
        #     return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        # data_collator = default_data_collator
        data_collator = DataCollatorForSeq2Seq(tokenizer)
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    # model.parallelize()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")

    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     predict_datasets = [predict_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         predict_datasets.append(raw_datasets["test_mismatched"])

    #     for predict_dataset, task in zip(predict_datasets, tasks):
    #         # Removing the `label` columns because it contains -1 and Trainer won't like that.
    #         predict_dataset = predict_dataset.remove_columns("label")
    #         predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    #         predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

    #         output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
    #         if trainer.is_world_process_zero():
    #             with open(output_predict_file, "w") as writer:
    #                 logger.info(f"***** Predict results {task} *****")
    #                 writer.write("index\tprediction\n")
    #                 for index, item in enumerate(predictions):
    #                     if is_regression:
    #                         writer.write(f"{index}\t{item:3.3f}\n")
    #                     else:
    #                         item = label_list[item]
    #                         writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
