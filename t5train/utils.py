task_to_labels = {
    "cola": ("not_acceptable", "acceptable"),
    # "mnli": None,
    "mrpc": ("not_equivalent", "equivalent"),
    "qnli": ("entailment", "not_entailment"),
    "qqp": ("not_duplicate", "duplicate"),
    "rte": ("entailment", "not_entailment"),
    "sst2": ("negative", "positive"),
    # "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}

def label2text(task_name, label):
    if task_to_labels[task_name] is None:
        return label
    else:
        return task_to_labels[task_name][label]
