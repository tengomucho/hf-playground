#!/usr/bin/python

# from https://twitter.com/reach_vb/status/1759716375033938291


from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torch
import time
import datetime
import os
import platform
from contextlib import contextmanager

os.environ["PJRT_DEVICE"] = "TPU"


# This will allow ignoring profiling on unsupported platforms
class DummyProfiler:
    @contextmanager
    def StepTrace(self, *args, **kwargs):
        yield

    @contextmanager
    def Trace(self, *args, **kwargs):
        yield

    def start_server(self, int):
        return

    def trace_detached(self, *args, **kwargs):
        return


xp = DummyProfiler()
if "CPU" in os.environ:
    device = "cpu"
else:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.profiler as xp

        device = "xla"
    except ModuleNotFoundError:
        device = "mps"


def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


def decode_one_tokens(
    model, cur_token, input_pos, cache_position, past_key_values, step
):
    with xp.StepTrace("decode_one_tokens", step_num=step):
        with xp.Trace("inference"):
            logits = model(
                cur_token,
                position_ids=input_pos,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )[0]
        with xp.Trace("sample_greedy"):
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
                "backend": "openxla",
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
tokenizer = AutoTokenizer.from_pretrained(
    model_id, return_tensors="pt", padding_side="right"
)

# prompts = ["Here's a funny thing:", "This is a good recipe for a dessert:"]
# # 4 prompts test, for a change
# prompts = ["Here's a funny thing:", "This is a good recipe for a dessert:", "Give me one more chance", ""]
prompts = ["Here's a funny thing:", "Here's a funny thing: I've been"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
batch_size, sequence_length = inputs["input_ids"].shape
max_cache_length = 1024
max_new_tokens = 20

profiler_port = 9874
profile_logdir = "./logdir/"
profile_duration_ms = 5000
profile_step = 4


def test_gemma(device, model):
    server = xp.start_server(profiler_port)
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=2,
        max_cache_len=max_cache_length,
        device=model.device,
        dtype=model.dtype,
    )
    start = time.time()
    cache_position = torch.arange(sequence_length, device=device)
    generated_ids = torch.zeros(
        (batch_size, sequence_length + max_new_tokens + 1),
        dtype=torch.int,
        device=device,
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch.int)

    # prefill here
    attention_mask = inputs["attention_mask"]
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    logits = model(
        **inputs,
        cache_position=cache_position,
        return_dict=False,
        use_cache=True,
        position_ids=pos_ids,
        past_key_values=past_key_values,
    )[0]
    next_token = sample_greedy(logits)
    if device == "xla":
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
        if i == profile_step:
            xp.trace_detached(
                f"localhost:{profiler_port}",
                profile_logdir,
                duration_ms=profile_duration_ms,
            )

        step_start = time.time()
        next_token = decode_one_tokens(
            model, next_token.clone(), pos_ids, cache_position, past_key_values, i
        )
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
    return generated_ids


with torch.no_grad():
    generated_ids = test_gemma(device, model)

print(f"Getting data back {datetime.datetime.now()}")

decoded_texts = tokenizer.batch_decode(generated_ids)
for i, text in enumerate(decoded_texts):
    print(i, text)

end = time.time()
print(
    f"Program run in {end - prg_start} seconds. Device: {device} System: {platform.system()}"
)
