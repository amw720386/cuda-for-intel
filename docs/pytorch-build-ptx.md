# Building PyTorch with PTX for the shim

PyTorch's own cu126/cu128 wheels (and every Windows wheel from cu121) carry SASS only, which the
shim cannot run (`docs/sass-support.md`). Building from source with `+PTX` keeps a virtual
architecture's PTX in every fatbin, and the shim translates that.

## Recipe

```
scripts/torch/build-torch-ptx.sh v2.9.1 /path/to/workdir     # ARCH_LIST defaults to "8.0+PTX"
```

Key choices:

* `TORCH_CUDA_ARCH_LIST="8.0+PTX"`: compute_80 PTX (ISA about 7.x/8.x) needs fewer features than the
  sm_120 PTX of the stock Linux cu129 wheels (no wgmma/tcgen05). Add real SASS entries only if you
  also want to run on NVIDIA hardware (`"7.5;8.0+PTX"`); the shim uses the PTX only.
* `USE_CUDNN=0`, `USE_NCCL=0`, `USE_DISTRIBUTED=0`: no stand-ins exist for these in the default
  configuration (cuDNN is optional via the `cudnn-fast` branch).
* `USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0`: SDPA uses the math path.
* Verify with `python scripts/ptxscan.py <libtorch_cuda>`: every fatbin must list a PTX arch.

Resources: about 64 GB RAM, 32 cores, 60 GB disk, 2-6 hours. The Windows/MSVC flavour uses the
same environment variables from a VS developer prompt (`python setup.py bdist_wheel`).

## Status

Not built yet. A build box is required (the Arc test machine has no CUDA toolkit and is shared);
no paid cloud compute has been launched. Needed: one Linux CPU box (c6i.8xlarge-class or larger,
about 200 GB disk, a few hours) or a Windows box with VS 2022 and the CUDA toolkit. Because the
current test host is Windows, a Windows build is the directly usable output; a Linux wheel needs a
Linux Intel host.
