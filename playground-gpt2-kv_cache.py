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


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Once upon a time"
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)


model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model = model.eval()

# Source for hte Key-value cache code:
# https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial_optimization.md#32-the-key-value-cache
past_key_values = None # past_key_values is the key-value cache
next_token_id = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
current_text = text

for _ in range(20):
  print(f"next_token_id={next_token_id}, text={tokenizer.batch_decode(next_token_id)}")
  next_logits, past_key_values = model(next_token_id,
                                       past_key_values=past_key_values,
                                       use_cache=True).to_tuple()
  next_logits = next_logits[:, -1:]
  next_token_id = torch.argmax(next_logits, dim=-1)

  print("shape of input_ids", next_token_id.shape)
  # past_key_values are of shape [num_layers, 0 for k, 1 for v, batch_size, length, hidden_dim]
  print("length of key-value cache", past_key_values[0][0].shape[-2])
  current_text += tokenizer.batch_decode(next_token_id)[0]

print(current_text)
