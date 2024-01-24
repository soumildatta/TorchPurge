import torch
import gc

# Compatible with CUDA and MPS devices
def torchPurge(tensors: list):
    for tensor in tensors:
        # Move tensor to cpu first then delete
        if tensor.is_cuda: 
            tensor = tensor.cpu()
        del tensor 

    # Clear the cache and run garbage collection
    torch.cuda.empty_cache() 
    return gc.collect()