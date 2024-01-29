from transformers import AutoTokenizer
import transformers
import torch

# You should login to hf for this:
# huggingface-cli login
# Tokens are here: https://huggingface.co/settings/tokens

# Resources:
# https://huggingface.co/blog/llama2
# https://huggingface.co/meta-llama/Llama-2-7b
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    raise RuntimeError("where's cuda? :( ")

model = "huggingface/llama-7b"

tokenizer = AutoTokenizer.from_pretrained(model,
                                          torch_dtype=torch.float16,
                                          device_map=device)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map=device,
)

sequences = pipeline(
    'Question: What song would Freddy Mercury write today if he were still alive?\nAnswer: ',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")