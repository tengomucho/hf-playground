from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
import os
os.environ['PJRT_DEVICE'] = 'TPU'

if 'CPU' in os.environ:
    device = 'cpu'
else:
    try:
        import torch_xla.core.xla_model as xm
        device = 'xla'
    except:
        device = 'mps'

# Resources:
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# I could have used the basic Mistral 7B, but this one seems easier to test out

model_name = "mistralai/Mistral-7B-Instruct-v0.1"


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             device_map=device)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)

model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])