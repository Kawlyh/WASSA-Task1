import os
from typing import Optional
import torch
import sys
import logging
import datasets
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import MSELoss
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding, \
    DebertaPreTrainedModel, DebertaForSequenceClassification, DebertaModel
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout

# 读取训练和验证数据集
train = pd.read_csv("/kaggle/input/wa-iri/WASSA23_essay_level_with_labels_train.tsv", delimiter="\t")
# train = pd.read_csv("corpus/WASSA23_essay_level_with_labels_train.tsv", delimiter="\t")
# print(train.columns)
val = pd.read_csv("/kaggle/input/wdata-emo/emo-full_eval.tsv")
# val = pd.read_csv("corpus/dev/emo-full_eval.tsv")
# print(val.columns)
test = pd.read_csv("/kaggle/input/wa-iri/WASSA23_essay_level_dev.tsv", delimiter="\t")
# test = pd.read_csv("corpus/dev/WASSA23_essay_level_dev.tsv", delimiter="\t")
# print(test.columns)
device = "cuda" if torch.cuda.is_available() else "cpu"

label2id = {'Anger': 0, 'Anger/Disgust': 1, 'Anger/Neutral': 2, 'Anger/Sadness': 3,
            'Disgust': 4, 'Disgust/Hope': 5, 'Disgust/Sadness': 6, 'Fear': 7,
            'Fear/Sadness': 8, 'Hope': 9, 'Hope/Neutral': 10, 'Hope/Sadness': 11,
            'Joy': 12, 'Neutral': 13, 'Neutral/Sadness': 14, 'Neutral/Surprise': 15,
            'Sadness': 16, 'Sadness/Surprise': 17, 'Anger/Disgust/Sadness': 18, 'Anger/Fear': 19,
            'Anger/Hope': 20, 'Anger/Joy': 21, 'Anger/Surprise': 22, 'Disgust/Fear': 23,
            'Disgust/Neutral': 24, 'Disgust/Surprise': 25, 'Fear/Hope': 26, 'Fear/Neutral': 27,
            'Joy/Neutral': 28, 'Joy/Sadness': 29, 'Surprise': 30
            }
id2label = {0: 'Anger', 1: 'Anger/Disgust', 2: 'Anger/Neutral', 3: 'Anger/Sadness',
            4: 'Disgust', 5: 'Disgust/Hope', 6: 'Disgust/Sadness', 7: 'Fear',
            8: 'Fear/Sadness', 9: 'Hope', 10: 'Hope/Neutral', 11: 'Hope/Sadness',
            12: 'Joy', 13: 'Neutral', 14: 'Neutral/Sadness', 15: 'Neutral/Surprise',
            16: 'Sadness', 17: 'Sadness/Surprise', 18: 'Anger/Disgust/Sadness', 19: 'Anger/Fear',
            20: 'Anger/Hope', 21: 'Anger/Joy', 22: 'Anger/Surprise', 23: 'Disgust/Fear',
            24: 'Disgust/Neutral', 25: 'Disgust/Surprise', 26: 'Fear/Hope', 27: 'Fear/Neutral',
            28: 'Joy/Neutral', 29: 'Joy/Sadness', 30: 'Surprise'}

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    temp1 = list(map(str, train['essay']))
    temp2 = list(map(str, val['essay']))
    temp3 = list(map(str, test['essay']))

    for i in range(len(train['emotion'])):
        train['emotion'][i] = label2id[train['emotion'][i]]

    for i in range(len(val['emotion'])):
        val['emotion'][i] = label2id[val['emotion'][i]]

    train_dict = {'labels': train['emotion'], 'text': temp1}
    val_dict = {'labels': val["emotion"], 'text': temp2}
    test_dict = {'text': temp3}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    # batch_size = 32
    model_id = "microsoft/deberta-v2-xxlarge"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=31).to(device)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric = evaluate.load("f1")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=1,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    jieguo = []
    for i in range(len(test_pred)):
        jieguo.append(id2label[test_pred[i]])

    print(jieguo)

    result_output = pd.DataFrame(data={"emotion": jieguo})
    result_output.to_csv("result/predictions_EMO.tsv", index=False, header=None, sep="\t")
    logging.info('result saved!')
