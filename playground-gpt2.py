from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


# Set the device to cuda or MPS if available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

if device == 'cpu':
    raise RuntimeError("device seems to be cpu... :( ")
device = 'cpu'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)

# from transformers import set_seed
# set_seed(42)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.eval()
model.to(device)
model.generation_config.update(do_sample=True, max_length=50)

output = model.generate(**encoded_input)
generated_sequence = output.cpu().numpy().tolist()
decoded = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(decoded[0])
