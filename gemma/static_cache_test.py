#!/usr/bin/python

# from https://twitter.com/reach_vb/status/1759716375033938291


from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torch
import time
import datetime
import os
import platform

os.environ["PJRT_DEVICE"] = "TPU"

if 'CPU' in os.environ:
    device ="cpu"
else:
    try:
        import torch_xla.core.xla_model as xm
        device = "xla"
    except ModuleNotFoundError:
        device = "mps"


def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


def decode_one_tokens(model, cur_token, input_pos, cache_position):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        return_dict=False,
        use_cache=True,
    )[0]
    new_token = sample_greedy(logits)
    return new_token


def conditional_compile(func):
    if "DBG_COMPILE" in os.environ:
        if device == "mps":
            compile_params = {
                "backend": "aot_eager",
                "fullgraph": True,
            }
        else:
            compile_params = {
                "backend": "openxla_eval",
                # "mode":"reduce-overhead",
                # "fullgraph": True
            }
        compiled = torch.compile(func, **compile_params)
        return compiled
    return func


prg_start = time.time()

model_id = "google/gemma-2b"
torch_dtype = torch.float16 if device == "mps" else torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch_dtype, device_map=device
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
# prompts = ["Here's a funny thing:", "This is a good recipe for a dessert:"]
# # 4 prompts test, for a change
# prompts = ["Here's a funny thing:", "This is a good recipe for a dessert:", "Give me one more chance", ""]
prompts = ["Here's a funny thing:", "Here's a funny thing: I've been"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
batch_size, sequence_length = inputs["input_ids"].shape
max_cache_length = 1024
max_new_tokens = 20

with torch.no_grad():

    start = time.time()
    model._setup_cache(StaticCache, batch_size, max_cache_len=max_cache_length)
    end = time.time()
    print(f"Model cache setup took {end - start} seconds.")
    start = time.time()
    cache_position = torch.arange(sequence_length, device=device)
    generated_ids = torch.zeros(
        (batch_size, sequence_length + max_new_tokens + 1),
        dtype=torch.int,
        device=device,
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch.int)

    # prefill here
    attention_mask = inputs['attention_mask']
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    logits = model(
        **inputs, cache_position=cache_position, return_dict=False, use_cache=True, position_ids=pos_ids
    )[0]
    next_token = sample_greedy(logits)
    if device == 'xla':
        xm.mark_step()
    generated_ids[:, sequence_length] = next_token[:, 0]
    end = time.time()
    print(f"Prefill took {end - start} seconds.")

    pos_ids = pos_ids.max(axis=-1)[0].unsqueeze(1) + 1

    # decode_one_tokens = conditional_compile(decode_one_tokens)
    model = conditional_compile(model)
    cache_position = torch.tensor([sequence_length + 1], device=device)
    start = time.time()
    for i in range(max_new_tokens):
        step_start = time.time()
        next_token = decode_one_tokens(model, next_token.clone(), pos_ids, cache_position)
        generated_ids[:, cache_position] = next_token

        cache_position += 1
        pos_ids += 1
        if device == "xla":
            xm.mark_step()
        step_end = time.time()
        print(f"Step {i} took {step_end - step_start} seconds.")
        # print(f" token: {next_token} text: {tokenizer.batch_decode(next_token)}")
    end = time.time()
    print(f"Eval took {end - start} seconds.")


print(f"Getting data back {datetime.datetime.now()}")

decoded_texts = tokenizer.batch_decode(generated_ids)
for i, text in enumerate(decoded_texts):
    print(i, text)

end = time.time()
print(f"Program run in {end - prg_start} seconds. Device: {device} System: {platform.system()}")
