import os
import sys
import logging
import datasets
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorWithPadding, RobertaTokenizerFast, \
    RobertaForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertTokenizerFast, \
    BertForSequenceClassification, AutoModelForSequenceClassification, DebertaV2Tokenizer
from transformers import Trainer, TrainingArguments
import evaluate

# 读取训练和验证数据集
train = pd.read_csv("/home/wangyukun/workspace/wassa/corpus/WASSA23_essay_level_with_labels_train.tsv", delimiter="\t")
val = pd.read_csv("/home/wangyukun/workspace/wassa/corpus/dev/emp-full_eval.tsv")
test = pd.read_csv("/home/wangyukun/workspace/wassa/corpus/dev/WASSA23_essay_level_dev.tsv", delimiter="\t")

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # log设置
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    temp1 = list(map(str, train['essay']))
    temp2 = list(map(str, val['essay']))
    temp3 = list(map(str, test['essay']))
    print(type(train["empathy"][0]))

    # 结果
    pre = []
    for i in range(2):
        if i == 0:
            train_dict = {'label': train["empathy"], 'text': temp1}
            val_dict = {'label': val["empathy"], 'text': temp2}
            test_dict = {'text': temp3}
        else:
            train_dict = {'label': train["distress"], 'text': temp1}
            val_dict = {'label': val["distress"], 'text': temp2}
            test_dict = {'text': temp3}

        train_dataset = datasets.Dataset.from_dict(train_dict)
        val_dataset = datasets.Dataset.from_dict(val_dict)
        test_dataset = datasets.Dataset.from_dict(test_dict)

        # 加载分词器
        # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
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

        # 加载模型
        # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        # model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
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
            per_device_train_batch_size=2,  # batch size per device during training
            per_device_eval_batch_size=1,  # batch size for evaluation
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

    result_output = pd.DataFrame(data={"empathy": pre[0], "distress": pre[1]})
    result_output.to_csv("/home/wangyukun/workspace/wassa/result/best-predictions_EMP.tsv", index=False, header=None, sep="\t")
    logging.info('result saved!')
