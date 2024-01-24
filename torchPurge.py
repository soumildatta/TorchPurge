import torch
import gc

# Compatible with CUDA and MPS devices
def torchPurge(mps_tensors: list):
    for tensor in mps_tensors:
        if tensor.is_cuda: 
            tensor = tensor.cpu()
        del tensor 

    # Clear the cache and run garbage collection
    torch.cuda.empty_cache() 
    return gc.collect()