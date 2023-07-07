import torch, mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We are running on - "+ str(device) +"-----!")

import os
os.environ["MLFLOW_EXPERIMENT_NAME"] = "AML-FINETUNE-ELEUTHERAI_GPT_NEOX_20B-LLM-V1"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
#os.environ["MLFLOW_TRACKING_URI"]=""
#os.environ["HF_MLFLOW_LOG_ARTIFACTS"]="1"

model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="eleutherai_gpt_neox_20b")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0},cache_dir="eleutherai_gpt_neox_20b")

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

with mlflow.start_run() as run:
    trainer.train()
    trainer.save_model("finetuned_eleutherai_gpt_neox_20b")
mlflow.end_run()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")
trainer.save_model("outputs")

lora_config = LoraConfig.from_pretrained("outputs")
model = get_peft_model(model, lora_config)

text = "Who is Satya Nadella?"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))