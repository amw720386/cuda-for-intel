#!/usr/bin/env bash
# Build PyTorch from source with PTX embedded, so the CUDA for Intel runtime (which translates
# PTX and never executes SASS) can run it. See docs/pytorch-build-ptx.md.
#
# Usage: scripts/torch/build-torch-ptx.sh [torch-tag] [work-dir]
#   env: ARCH_LIST (default "8.0+PTX"), MAX_JOBS (default: nproc), PYTHON (default python3)
# Needs: CUDA toolkit (nvcc) on PATH, cmake, ninja, a C++17 compiler, ~64 GB RAM / 32 cores
# recommended, ~60 GB disk. Does not touch system locations; everything lives in work-dir.
set -euo pipefail
TAG="${1:-v2.9.1}"
WORK="${2:-$PWD/torch-ptx-build}"
ARCH_LIST="${ARCH_LIST:-8.0+PTX}"
PY="${PYTHON:-python3}"
export MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}"

command -v nvcc >/dev/null || { echo "nvcc not found: install a CUDA toolkit (12.x)" >&2; exit 1; }
mkdir -p "$WORK"; cd "$WORK"
[ -d pytorch ] || git clone --branch "$TAG" --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/pytorch/pytorch
"$PY" -m venv venv; . venv/bin/activate
pip install -q -r pytorch/requirements.txt wheel

cd pytorch
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"   # "+PTX" on the last entry keeps compute_80 PTX in every fatbin
export USE_CUDA=1 USE_CUDNN=0 USE_NCCL=0 USE_DISTRIBUTED=0 USE_FLASH_ATTENTION=0 \
       USE_MEM_EFF_ATTENTION=0 USE_KINETO=0 BUILD_TEST=0 USE_MKLDNN=1
# cuDNN off: the shim has no cuDNN backend (torch.backends.cudnn.enabled=False at run time).
# Flash/mem-efficient attention off: their cutlass kernels are sm80 tensor-core heavy; SDPA
# falls back to the math path, which the shim runs.
"$PY" setup.py bdist_wheel
echo "wheel: $(ls "$PWD"/dist/*.whl)"

# Verify PTX is present before shipping the wheel anywhere.
cd "$WORK"
"$PY" "${CFI_REPO:-$OLDPWD}/scripts/ptxscan.py" venv/lib/python*/site-packages/torch/lib/libtorch_cuda.so 2>/dev/null \
  || echo "run scripts/ptxscan.py on the installed libtorch_cuda to confirm PTX entries"
