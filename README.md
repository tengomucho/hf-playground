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

# Prepare an env for Pytorch/XLA


If using Ubuntu 22.04, `nvidia-cuda-toolkit` is too old. Follow instructions at https://developer.nvidia.com/cuda-12-1-0-download-archive to get the latest cuda package.


```sh
python -m venv ~/Dev/venv/hf-xla
source ~/Dev/venv/hf-xla/bin/activate
pip install transformers accelerate
# For pytorch
pip install torch
# For pytorch_xla, take the package with installed CUDA, see https://github.com/pytorch/xla/blob/master/docs/gpu.md
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.1.0-cp310-cp310-manylinux_2_28_x86_64.whl

```

Once you've done that, it is possible to verify that the XLA device works correctly:

```python
import os
os.environ['PJRT_DEVICE']='GPU'
import torch
import torch_xla.core.xla_model as xm

t = torch.randn(2, 2, device=xm.xla_device())
print(t)
```

It is possible to modify other tests and set the `PJRT_DEVICE` environment variable and set the device: `device = xm.xla_device()` to perform the inference (tested only with distilbert, other tests might need more changes).

# Prepare a simple venv for Flax/JAX with CUDA

Make sure you are using a recent CUDA toolkit as mentioned before. To install Flax and JAX:

```sh
python -m venv ~/Dev/venv/hf-flax
source ~/Dev/venv/hf-flax/bin/activate
pip install transformers accelerate
# Install JAX with cuda
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Install flax
pip install flax
```
