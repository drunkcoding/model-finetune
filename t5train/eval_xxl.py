from typing import List
import requests
import os
from dataclasses import dataclass, field
from functools import partial
import re
import time
import numpy as np

import torch
from datasets import concatenate_datasets, load_dataset, load_metric
from hfutils.loader import load_glue_val, t5_preprocess_function
from hfutils.pipe.t5 import T5DeepSpeedPipe
from hfutils.logger import Logger
from hfutils.pipe.base import get_num_layers
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from transformers.models.t5.configuration_t5 import T5Config

from hfutils.constants import TASK_TO_KEYS, TASK_TO_LABELS

def label2text(task_name, label):
    if TASK_TO_LABELS[task_name] is None:
        return label
    else:
        return TASK_TO_LABELS[task_name][label]

def token2label(tokens, label_tokens: List):
    return [label_tokens.index(t) for t in tokens]

import deepspeed
import logging
from deepspeed.utils import RepeatingLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_EXTENSIONS_DIR"] = "."


@dataclass
class Arguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    deepspeed_config: str = field(
        default=None, metadata={"help": "DeepSpeed configuration path."},
    )
    local_rank: int = field(
        default=-1, metadata={"help": "Place holder for deepspeed launcher."},
    )


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

logger = Logger(__file__, logging.INFO, 50000000, 5)
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

if local_rank == 0:
    logger.info("=================================")
    logger.info("%s", args)

tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
config = T5Config.from_pretrained(args.model_name_or_path)

preprocess_function = partial(
    t5_preprocess_function, tokenizer=tokenizer, padding="max_length", max_length=128,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
dataset = load_glue_val(preprocess_function).shuffle()


# print(eval_dataset[0])
batch_size = 1

def eval_generator():
    dataloader = DataLoader(
        dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size, drop_last=True,
    )
    pid = os.getpid()

    for batch in tqdm(dataloader, desc=f"{pid}-eval_generator"):
        shape = batch["input_ids"].shape
        yield (
            (batch["input_ids"], batch["attention_mask"],),
            torch.zeros(shape[0]),
            batch['labels']
        )


def get_energy_by_group():
    response = requests.get("http://localhost:8002/metrics")
    text = response.text
    energy_groups = re.findall(
        r'nv_energy_consumption{gpu_uuid="(.*)"} (\d+.\d+)', text
    )
    energy_groups = dict(energy_groups)
    for k in energy_groups:
        energy_groups[k] = float(energy_groups[k])
    return energy_groups

label_tokens = [
    tokenizer(label, max_length=2).input_ids[0]
    for label in TASK_TO_LABELS['mnli']
    if label is not None
]


deepspeed.init_distributed()
# model = T5DeepSpeedPipe(config, num_stages=torch.cuda.device_count())
model = T5DeepSpeedPipe(config, num_stages=world_size)

engine, _, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model)

start_energy = sum(list(get_energy_by_group().values()))
inference_count = 0
logits_list = []
labels_list = []
metric = load_metric("accuracy")
for step, batch in enumerate(RepeatingLoader(eval_generator())):
    labels = batch[-1].detach().cpu().numpy()
    batch = batch[:-1]
    start_time = time.perf_counter()
    outputs = engine.eval_batch(iter([batch] * 1), compute_loss=False)
    end_time = time.perf_counter()
    if outputs != None:
        inference_count += 1
        print(outputs.shape, labels.shape)
        logits = outputs.squeeze(1)[:, label_tokens]
        predictions = np.argmax(logits.detach().cpu(), axis=-1)
        labels = token2label(labels[:, 0].flatten(), label_tokens)
        metric.add_batch(
            predictions=predictions,
            references=labels
        )
    if local_rank == 0:
        logger.info(
            "(%s) start_time %s, end_time %s, diff %s",
            world_size,
            start_time,
            end_time,
            end_time - start_time,
        )
logger.info("%s %s ", os.getpid(), inference_count)
if inference_count > 0:
    logger.info("Metric %s ", metric.compute())

end_energy = sum(list(get_energy_by_group().values()))
if local_rank == 0:
    logger.info(
        "(%s) start_energy %s, end_energy %s, diff %s",
        world_size,
        start_energy,
        end_energy,
        end_energy - start_energy,
    )

while True:
    pass