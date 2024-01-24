# TorchPurge

A straightforward and simple little tool to ensure pytorch tensors have been purged from the GPU in case of unexpected interrupts. Compatible with CUDA or MPS devices.

## Usage
Call the `torchPurge()` function with the names of your tensor variables as a list as the argument. This will clear the cache and remove the tensors from the GPU if it exists. For example
```py
torchPurge([tensor1, tensor2, ...])
```
