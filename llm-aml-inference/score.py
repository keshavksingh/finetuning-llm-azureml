import os
import logging
import json
import numpy
import joblib


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global tokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "outputs"
    )
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
    lora_config = LoraConfig.from_pretrained(model_path)
    model = get_peft_model(model, lora_config).to(device)
    
    logging.info("Init complete")


def run(input: str):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    The function takes query and the max response tokens as inputs.
    The method returns the response from the fine-tuned LLM.
    Expected Input in shape input = {
                                      "input":"What is the capital of Germany?",
                                       "max_token_number":10
                                    }
    """
    logging.info("model 1: request received")
    data = json.loads(input)
    input_data= data["input"]
    max_new_tokens= data["max_token_number"]
    print("Input String - "+str(input_data))
    print("Input max_new_tokens - "+str(max_new_tokens))
    device = "cuda:0"
    inputs = tokenizer(input_data, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.replace("\n\r", "").replace("\n", "").replace("\r", "")
    result_json = json.dumps({"result":str(generated_text)})
    print("Output Response String - !")
    print(result_json)
    logging.info("Request processed")
    return result_json