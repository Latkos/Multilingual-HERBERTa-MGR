"""Singleton class definition and basic ner model configuration"""

from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification


class Singleton(type):
    """Metaclass with the Singleton pattern"""

    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]


class NERConfig(metaclass=Singleton):
    DICT_LABELS = {
        "O": 0,
        "B-ENTITIY_1": 1,
        "I-ENTITIY_1": 2,
        "B-ENTITIY_2": 3,
        "I-ENTITIY_2": 4,
    }
    LABEL_NAMES = list(DICT_LABELS.keys())
    MODEL_NAME = "bert-base-multilingual-cased"
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    METRIC = load_metric("seqeval")
    DATA_COLLATOR = DataCollatorForTokenClassification(TOKENIZER)
