from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
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

model_id = "openai-community/gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_id)
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)

# from transformers import set_seed
# set_seed(42)
start = time.time()
model = GPT2LMHeadModel.from_pretrained(model_id)
model.to(device)
model = model.eval()
model.generation_config.update(do_sample=True, max_length=50)
end = time.time()
print(f"Setup took {end - start} seconds.")

start = time.time()
output = model.generate(**encoded_input, pad_token_id=tokenizer.eos_token_id)
end = time.time()
print(f"Model generate took {end - start} seconds.")

if device == 'xla':
    xm.mark_step()
generated_sequence = output.cpu().numpy().tolist()
decoded = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(decoded[0])
