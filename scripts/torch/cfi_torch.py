"""PyTorch glue for the cuda-for-intel cuDNN shim: shim ON by default, unsupported calls routed to ATen.

Usage (after the shim DLLs are staged, see docs/pytorch.md), once per process, right after `import torch`:

    import cfi_torch; cfi_torch.install()

* Default: `torch.backends.cudnn.enabled = True` so 2-D convolutions (fp32/fp16/bf16, forward, backward data,
  backward filter, any stride/dilation/groups/padding, NCHW or channels_last) and batch norm run on the shim.
* `CFI_CUDNN=0` (or `install(enable=False)`): cuDNN stays off and ATen's own kernels run everything.
* The shim answers CUDNN_STATUS_NOT_SUPPORTED for what it has no kernel for (3-D convolution, fp64 convolution,
  cuDNN RNN/LSTM/GRU, CTC loss). PyTorch would raise on that, so these calls are wrapped to run with cuDNN
  disabled (forward and the backward node), exactly like before the shim existed.
  Code that bypasses torch.nn.functional / nn.RNN modules (raw torch.conv3d, torch._VF.lstm) is not wrapped.
"""
import functools
import os

_installed = False
_state = {"stack": []}


def _enabled_default():
    return os.environ.get("CFI_CUDNN", "1").strip().lower() not in ("0", "off", "false", "no")


def _pin_backward(node):
    """Run this autograd node (and only it) with cuDNN off: the backend is re-selected at backward time."""
    import torch

    def pre(_grad_outputs):
        _state["stack"].append(torch.backends.cudnn.enabled)
        torch.backends.cudnn.enabled = False

    def post(_grad_inputs, _grad_outputs):
        if _state["stack"]:
            torch.backends.cudnn.enabled = _state["stack"].pop()

    node.register_prehook(pre)
    node.register_hook(post)


def _wrap_conv(fn, unsupported):
    import torch

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        x = args[0] if args else kwargs.get("input")
        if not (isinstance(x, torch.Tensor) and x.is_cuda and torch.backends.cudnn.enabled and unsupported(x, args, kwargs)):
            return fn(*args, **kwargs)
        with torch.backends.cudnn.flags(enabled=False):
            out = fn(*args, **kwargs)
        node = getattr(out, "grad_fn", None)
        if node is not None:
            _pin_backward(node)
        return out

    wrapper._cfi_wrapped = True
    return wrapper


def _wrap_forward_off(fn):
    import torch

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return fn(*args, **kwargs)

    wrapper._cfi_wrapped = True
    return wrapper


def install(enable=None):
    """Enable the shim path (default on) and wrap the calls it cannot run. Idempotent."""
    global _installed
    import torch
    import torch.nn.functional as F
    from torch.nn.modules import rnn

    on = _enabled_default() if enable is None else bool(enable)
    torch.backends.cudnn.enabled = on
    if not on or _installed:
        return on
    _installed = True
    is64 = lambda x, a, k: x.dtype == torch.float64
    is3d = lambda x, a, k: x.dim() == 5 or x.dtype == torch.float64
    for name, pred in (("conv1d", is64), ("conv2d", is64), ("conv_transpose1d", is64), ("conv_transpose2d", is64),
                       ("conv3d", is3d), ("conv_transpose3d", is3d)):
        fn = getattr(F, name)
        if not getattr(fn, "_cfi_wrapped", False):
            setattr(F, name, _wrap_conv(fn, pred))
    for cls in (rnn.RNN, rnn.LSTM, rnn.GRU):
        if not getattr(cls.forward, "_cfi_wrapped", False):
            cls.forward = _wrap_forward_off(cls.forward)
    if not getattr(F.ctc_loss, "_cfi_wrapped", False):
        F.ctc_loss = _wrap_forward_off(F.ctc_loss)
    return on
