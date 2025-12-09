import csv
import os

import pandas as pd
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

from src.dataset.SRE import SREDataset

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "https://hf-mirror.com"
)

#准备训练数据
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

def load_sentencere_factoring():
    folder_path = "/media/zhang/orico/idata/datasets/sentence_refactoring"

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    examples = []
    for file in csv_files:
        with open(os.path.join(folder_path, file), 'r') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=';')
            print(f"Reading {file}:")
            for row in csv_reader:
                example = {}
                row_list = row# ace with your processing logic
                example["instruction"] = row_list[0]
                example["input"] = ""
                example["output"] = row_list[1]
                examples.append(example)
                print(row_list)
            print()  # Add an empty line for separation
    return examples

#hugging face数据集路径
dataset = load_dataset("kigner/ruozhiba-llama3", split = "train")
# dataset = load_sentencere_factoring()
# dataset = dataset.map(formatting_prompts_func, batched = True,)

# dataset = SREDataset(folder_path="/media/zhang/orico/idata/datasets/sentence_refactoring")

dataset = Dataset.from_pandas(pd.DataFrame(data=load_sentencere_factoring()))
dataset = dataset.map(formatting_prompts_func, batched = True,)

#设置训练参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,
    loftq_config = None,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    ),
)
#开始训练
trainer.train()

#保存微调模型
model.save_pretrained("lora_model")

#合并模型，保存为16位hf
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)

#合并模型，并量化成4位gguf
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
