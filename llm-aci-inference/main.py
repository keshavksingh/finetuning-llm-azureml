# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI
#from dotenv import load_dotenv
import os
#load_dotenv() 
# Initialize the FastAPI application
app = FastAPI()
## One Time Model Loading
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

class generativeai:
    def __init__(self, input,max_new_tokens):
        """
        To initalize the Details
        """
        self.input = input
        self.max_new_tokens=max_new_tokens

    def generate_response(self):
        text = self.input
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return the Response
        return generated_text

# Create the POST endpoint with path '/predict'
@app.post("/generate")
async def generativellm(input: str,max_new_tokens:int):
    llmresponse =  generativeai(input,max_new_tokens)
    return {
        str(llmresponse.generate_response())
    }

if __name__ == '__main__':
    app.run(debug=True)