import os
os.environ["HF_TOKEN"] = "<HF Token>"
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

from transformers import AutoTokenizer, AutoModel
model_id = "keshavsingh/finetuned_eleutherai_gpt_neox_20b"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", cache_dir="eleutherai_gpt_neox_20b")
model = AutoModel.from_pretrained(model_id)

text = "Who is Satya Nadella?"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))