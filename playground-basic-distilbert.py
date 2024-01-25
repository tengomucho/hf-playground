from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Set the device to cuda or MPS if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'cpu':
    raise RuntimeError("where's cuda? :( ")
print(device)

raw_inputs = [
    "I've been waiting for this moment my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
inputs.to(device)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.to(device)
outputs = model(**inputs)


predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)