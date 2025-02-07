import os
from transformers import TrainingArguments, AutoTokenizer, HfArgumentParser
from utils.my_trainer import CustomTrainer
from utils.utils import my_compute_metrics,seed_everything,preprocess_logits_for_metrics
from typing import Optional
from dataclasses import dataclass, field
from model.my_model import MyCustomModel
from peft import LoraConfig
from datasets import load_dataset
from utils.data_collator import MyDataCollatorForLanguageModeling
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    # system config
    gpu: Optional[str] = field(default="6,7", metadata={"help": "gpu"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed"})

    # model
    llm_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "meta-llama/Meta-Llama-3-8B-Instruct，meta-llama/Meta-Llama-3.1-8B-Instruct "})
    
    # data
    select_data_num: Optional[int] = field(default=-100, metadata={"help": "the number of training data， -1 mean use all data"})
    dataset_name: Optional[str] = field(default="/raid/hpc/hekai/WorkShop/My_project/Score_LLM/data/human_annotation_data_2_out_prompt_v2.jsonl", metadata={"help": "data "})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    
    # log and save model
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    save_steps: Optional[int] = field(default=100, metadata={"help": "Number of updates steps before two checkpoint saves"})
    
    llm_requires_grad: Optional[bool] = field(default=True, metadata={"help": "True or  False"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "True or  /output/checkpoint-1400"})
    
    # training hypterparam
    learning_rate: Optional[float] = field(default=2.0e-5, metadata={"help": "the learning rate"})
    train_batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    max_seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "the number of gradient accumulation steps"})
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "the number of training epochs"})
        
    # eval
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "epoch, step"})
    eval_steps: Optional[int] = field(default=100000, metadata={"help": "eval_steps"})
    eval_accumulation_steps: Optional[int] = field(default=10, metadata={"help": "eval_accumulation_steps"})
    
    # unused
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="mistral-7b-finetuned-ultrachat", metadata={"help": "The name of the model on HF Hub"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
seed_everything(script_args.seed)

# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_MODE"] = "online"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = script_args.gpu



# set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.llm_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = 'left'
tokenizer.model_max_length = script_args.max_seq_length


def formatting_func(examples):
    question = examples["question"]
    answer = examples["answer"]
    text = f"{tokenizer.bos_token} Question : {question} {tokenizer.eos_token} " + f"{tokenizer.bos_token} Answer : {answer} {tokenizer.eos_token}"
    examples["text"] = text
    return examples

if script_args.select_data_num>0:
    split_text = "train[:{}]".format(script_args.select_data_num)
else:
    split_text = "train"
    


dataset = load_dataset('json', data_files=script_args.dataset_name, split=split_text)
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.map(formatting_func, num_proc=4, remove_columns=["question", "answer"])
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model = MyCustomModel(script_args.llm_requires_grad, 
                      script_args.load_in_8bit, 
                      script_args.load_in_4bit, 
                      script_args.llm_name, 
                      script_args.trust_remote_code, 
                      script_args.token, 
                      tokenizer,
                      )


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.train_batch_size,
    per_device_eval_batch_size=script_args.eval_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    # gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    # lr_scheduler_type="cosine",
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    bf16=True,
    warmup_ratio=0.1,
    evaluation_strategy=script_args.evaluation_strategy,
    # eval_accumulation_steps=script_args.eval_accumulation_steps,
    eval_steps=script_args.eval_steps,
    logging_first_step=True,
    remove_unused_columns=False,
    label_names=["labels"]
)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None


data_collator = MyDataCollatorForLanguageModeling(tokenizer)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.max_seq_length,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=my_compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)


trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)






