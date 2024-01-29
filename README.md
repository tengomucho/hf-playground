# HF Playground

These are my own scripts to test HuggingFace models. You can see it as a noob repository to check out if/how to use some models.

Note that I used 2 different venvs, one for Pytorch and one for flax (for the tests that end with -flax).

# Prepare a simple venv for Pytorch/CUDA

```sh
python -m venv ~/Dev/venv/hf
source ~/Dev/venv/hf/bin/activate
pip install transformers accelerate
# For pytorch/cuda(triton):
pip install torch torchvision torchaudio
```
