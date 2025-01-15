from transformers import AutoTokenizer
import transformers
import torch

device = 'mps'
model = "Qwen/Qwen2.5-0.5B-Instruct"

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
    'What is Deep Learning?',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")