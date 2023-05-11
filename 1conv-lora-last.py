import os

import peft
import torch
import sys
import logging
import datasets
# import evaluate

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, MODEL_TYPE_TO_PEFT_MODEL_MAPPING

train = pd.read_csv("corpus/last.tsv")
val = pd.read_csv("corpus/dev/conv-full_eval.tsv")
test = pd.read_csv("corpus/dev/WASSA23_conv_level_dev.tsv", delimiter="\t")

device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 3e-5
lr = 5e-5
lr = 7e-5

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    temp1 = list(map(str, train['0']))
    temp2 = list(map(str, val['text']))
    temp3 = list(map(str, test['text']))

    pre = []
    for i in range(3):
        if i == 0:
            train_dict = {'labels': train['1'], 'text': temp1}
            val_dict = {'labels': val["EmotionalPolarity"], 'text': temp2}
            test_dict = {'text': temp3}
        elif i == 1:
            train_dict = {'labels': train['2'], 'text': temp1}
            val_dict = {'labels': val["Emotion"], 'text': temp2}
            test_dict = {'text': temp3}
        else:
            train_dict = {'labels': train['3'], 'text': temp1}
            val_dict = {'labels': val["Empathy"], 'text': temp2}
            test_dict = {'text': temp3}

        train_dataset = datasets.Dataset.from_dict(train_dict)
        val_dataset = datasets.Dataset.from_dict(val_dict)
        test_dataset = datasets.Dataset.from_dict(test_dict)

        batch_size = 32
        model_id = "microsoft/deberta-v2-xxlarge"
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


        def preprocess_function(examples):
            return tokenizer(examples['text'], truncation=True)


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

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # metric = evaluate.load("pearsonr")

        # def compute_metrics(eval_pred):
        #     logits, labels = eval_pred
        #     predictions = np.argmax(logits, axis=-1)
        #     return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir='./checkpoint',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=2,  # batch size per device during training
            per_device_eval_batch_size=1,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=100,
            save_strategy="no",
            # evaluation_strategy="epoch"
            learning_rate=lr,
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=tokenized_train,  # training dataset
            eval_dataset=tokenized_val,  # evaluation dataset
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
        )

        trainer.train()
        prediction_outputs = trainer.predict(tokenized_test)
        test_pred = prediction_outputs.predictions[:, -1]
        print(test_pred)
        pre.append(test_pred)

    result_output = pd.DataFrame(data={"EmotionalPolarity": pre[0], "Emotion": pre[1], "Empathy": pre[2]})
    result_output.to_csv(f"result/lora-predictions_CONV{lr}.tsv", index=False, header=None, sep="\t")
    logging.info(f'result saved!')