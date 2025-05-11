from transformers import AutoTokenizer, AutoModelForCausalLM
import os

models = {"Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf"}

base_dir = "./ckpts"

modeltype2path = {}

for name, model_id in models.items():
    local_path = os.path.join(base_dir, name)
    print(f"Downloading {name} from {model_id}...")
    
    os.makedirs(local_path, exist_ok=True)
    
    #tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_path)
    
    #model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(local_path)
    
    modeltype2path[name] = local_path

print("Done")
