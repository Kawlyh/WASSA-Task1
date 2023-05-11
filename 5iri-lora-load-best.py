import os
import evaluate
import peft
import torch
import sys
import logging
import datasets
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, MODEL_TYPE_TO_PEFT_MODEL_MAPPING

# 读取训练和验证数据集
train = pd.read_csv("corpus/WASSA23_essay_level_with_labels_train.tsv", delimiter="\t")
val = pd.read_csv("corpus/dev/iri-full_eval.tsv")
test = pd.read_csv("corpus/dev/WASSA23_essay_level_dev.tsv", delimiter="\t")

# train = pd.read_csv("/kaggle/input/wa-iri/WASSA23_essay_level_with_labels_train.tsv", delimiter="\t")
# val = pd.read_csv("/kaggle/input/per-data/per-full_eval.tsv")
# test = pd.read_csv("/kaggle/input/wa-iri/WASSA23_essay_level_dev.tsv", delimiter="\t")


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    for e in range(len(train["taking"])):
        try:
            train["taking"][e] = float(train["taking"][e])
            train["distress"][e] = float(train["distress"][e])
            train["fantasy"][e] = float(train["fantasy"][e])
            train["concern"][e] = float(train["concern"][e])
        except:
            continue

    pre = []
    for i in range(4):
        if i == 0:
            train_dict = {'label': train["taking"], 'text': train['essay']}
            val_dict = {'label': val["taking"], 'text': val['essay']}
            test_dict = {'text': test['essay']}
        elif i == 1:
            train_dict = {'label': train["distress"], 'text': train['essay']}
            val_dict = {'label': val["distress"], 'text': val['essay']}
            test_dict = {'text': test['essay']}
        elif i == 2:
            train_dict = {'label': train["fantasy"], 'text': train['essay']}
            val_dict = {'label': val["Fantasy"], 'text': val['essay']}
            test_dict = {'text': test['essay']}
        else:
            train_dict = {'label': train["concern"], 'text': train['essay']}
            val_dict = {'label': val["concern"], 'text': val['essay']}
            test_dict = {'text': test['essay']}

        train_dataset = datasets.Dataset.from_dict(train_dict)
        val_dataset = datasets.Dataset.from_dict(val_dict)
        test_dataset = datasets.Dataset.from_dict(test_dict)


        model_id = "microsoft/deberta-v2-xxlarge"
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


        # 预处理
        def preprocess_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True)


        # 初始化分词器
        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_val = val_dataset.map(preprocess_function, batched=True)
        tokenized_test = test_dataset.map(preprocess_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1).to(device)

        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            # target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        # prepare int-8 model for training
        # model = prepare_model_for_int8_training(model)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 加载评估方法
        metric = evaluate.load("pearsonr")


        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir=f'/home/wangyukun/workspace/wassa/checkpoint{i}',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=16,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=100,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="pearsonr",
            greater_is_better=True
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
        test_pred = prediction_outputs.predictions[:, -1]
        # print(test_pred)
        pre.append(test_pred)


    result_output = pd.DataFrame(data={"Perspective-taking": pre[0], "Personal distress": pre[1], "Fantasy": pre[2],
                                       "Empathatic concern": pre[3]})
    result_output.to_csv("result/best-predictions_IRI.tsv", index=False, header=None, sep="\t")
    logging.info('result saved!')