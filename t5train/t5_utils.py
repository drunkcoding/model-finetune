from transformers import EvalPrediction, T5Tokenizer
import numpy as np

from hfutils.constants import TASK_TO_KEYS, TASK_TO_LABELS
from hfutils.arg_parser import HfArguments


def preprocess_function(examples, args: HfArguments, tokenizer: T5Tokenizer):

        data_args = args.data_args
        sentence1_key, sentence2_key = TASK_TO_KEYS[data_args.task_name]

        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Tokenize the texts
        sentence1_examples = examples[sentence1_key]
        sentence2_examples = None if sentence2_key is None else examples[sentence2_key]
        processed_examples = []
        for i in range(len(sentence1_examples)):
            elements = [
                    data_args.task_name, 
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
        result = tokenizer(*texts, padding=padding, max_length=data_args.max_length, truncation=True, return_tensors="np")

        if "label" in examples:

            labels = examples["label"]
            labels = [TASK_TO_LABELS[data_args.task_name][label] for label in labels]

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

def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)

        preds = [label_tokens.index(p) for p in preds[:, 0]] 
        label_ids = [label_tokens.index(p) for p in p.label_ids[:, 0]] 

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