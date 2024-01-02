from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id= "mistralai/Mistral-7B-v0.1"  # input: base model
adapter_path="out/checkpoint-663"       # input: adapters
save_to="models/Mistral-7B-finetuned"       # out: merged model ready for inference

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Add/set tokens same tokens to base model before merging, like we did before starting training https://github.com/geronimi73/qlora-minimal/blob/main/qlora.py#L27  
tokenizer.pad_token = "</s>"
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.eos_token_id = tokenizer.eos_token_id

# Load LoRA and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)
