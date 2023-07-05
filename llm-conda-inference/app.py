import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from fastapi import FastAPI
from pydantic import BaseModel
from azureml.core.conda_dependencies import CondaDependencies

app = FastAPI()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We are running on - "+ str(device) +"!")

model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

device = "cuda:0"
lora_config = LoraConfig.from_pretrained("output")
model = get_peft_model(model, lora_config).to(device)

class InputText(BaseModel):
    text: str

class OutputText(BaseModel):
    generated_text: str

@app.post("/generate", response_model=OutputText)
def generate(input_text: InputText):
    text = input_text.text

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return OutputText(generated_text=generated_text)

if __name__ == "__main__":
    # Set up conda dependencies
    conda_dep = CondaDependencies()
    conda_dep.add_channel('default')
    conda_dep.add_channel('conda-forge')
    conda_dep.add_channel('pytorch')
    conda_dep.add_channel('nvidia')

    conda_dep.add_conda_package('python=3.8.2')
    conda_dep.add_conda_package('pip=20.2.4')
    conda_dep.add_conda_package('pytorch=2.0.1')
    conda_dep.add_conda_package('torchvision=0.15.2')
    conda_dep.add_conda_package('torchaudio=2.0.2')
    conda_dep.add_conda_package('pytorch-cuda=11.7')

    conda_dep.add_pip_package('bitsandbytes==0.39.1')
    conda_dep.add_pip_package('git+https://github.com/huggingface/transformers.git')
    conda_dep.add_pip_package('git+https://github.com/huggingface/peft.git')
    conda_dep.add_pip_package('git+https://github.com/huggingface/accelerate.git')
    conda_dep.add_pip_package('datasets')
    conda_dep.add_pip_package('scipy')
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True, extra_files=['./output'], extra_dirs=['./output'], log_level="info", conda_env=conda_dep)
