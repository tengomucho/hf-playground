from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
from copy import deepcopy

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
text = "Once upon a time"
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)


model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model = model.eval()

# Source for hte Key-value cache code:
# https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial_optimization.md#32-the-key-value-cache
past_key_values = None # past_key_values is the key-value cache

num_new_tokens = 12

# The argument, by default 3 is the padding number
pad_num = 3
if len(sys.argv) > 1:
    pad_num = int(sys.argv[1])

next_token_id = tokenizer(text, return_tensors="pt")["input_ids"]
next_token_id0 = next_token_id
if pad_num > 0:
    next_token_id0 = model.generate(input_ids=next_token_id, pad_token_id=tokenizer.eos_token_id, max_new_tokens=pad_num)
input_len = next_token_id0.shape[-1]

next_token_id1 = torch.cat([torch.Tensor([tokenizer.eos_token_id] * pad_num), next_token_id[0]]).expand(1, -1)
next_token_id1 = next_token_id1.to(torch.int64)

next_token_id = torch.cat([next_token_id1, next_token_id0])
print("next_token_id")
print(next_token_id)

attention_mask = torch.tensor([[0] * pad_num + [1, 1, 1, 1], [1] * input_len]).to(device)
position_ids = attention_mask.cumsum(-1) - 1
position_ids = position_ids.masked_fill(attention_mask == 0, 0)
print("position_ids")
print(position_ids)
print("attention_mask")
print(attention_mask)

# output = model.generate(input_ids=next_token_id,
#                         attention_mask=attention_mask,
#                         position_ids=position_ids,
#                         pad_token_id=tokenizer.eos_token_id,
#                         max_new_tokens=num_new_tokens)
# print(output)
# decoded = tokenizer.batch_decode(output.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print(decoded)

# exit()

# This part is to do the same but token by token
print("-----")

generated_tokens = deepcopy(next_token_id)

offset = torch.max(position_ids)

for _ in range(num_new_tokens):
    print(f"next_token_id={next_token_id}, text={tokenizer.batch_decode(next_token_id)} att_mask={attention_mask}, pos_ids={position_ids}")
    breakpoint()
    next_logits, past_key_values = model(next_token_id,
                                         past_key_values=past_key_values,
                                         attention_mask=attention_mask,
                                         position_ids=position_ids,
                                         use_cache=True).to_tuple()


    next_logits = next_logits[:, -1:]
    next_token_id = torch.argmax(next_logits, dim=-1)
    # attention_mask = torch.ones([2, 1], dtype=torch.int64)
    attention_mask = torch.cat((attention_mask, torch.ones([2, 1], dtype=torch.int64)), dim=-1)
    # breakpoint()
    position_ids = (position_ids.max(axis=-1)[0] + 1).reshape(-1, 1)
    print(position_ids.shape)
    # past_key_values are of shape [num_layers, batch_size, length, hidden_dim]
    print("length of key-value cache", past_key_values[0][0].shape[-2])
    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)


# print(current_text)
output = generated_tokens
decoded = tokenizer.batch_decode(output.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(decoded)

