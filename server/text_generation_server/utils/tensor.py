import torch

def clean_tensor(k, t):
    if t.untyped_storage().size() != t.element_size()*t.numel():
        #print("Dropping extra storage of tensor with key %s" % (k))
        t.set_(t.clone(memory_format=torch.contiguous_format))
        return
    strides = list(t.stride())
    if strides != sorted(strides, reverse=True):
        #print("Making tensor with key %s contiguous" % (k))
        t.set_(t.clone(memory_format=torch.contiguous_format))

def clean_attribute(k, el):
    if isinstance(el, torch.Tensor):
        clean_tensor(k, el)
    elif isinstance(el, list) or isinstance(el, tuple):
        for v in el:
            clean_attribute(k, v)

def clean_batch(batch):
    for k, v in batch.__dict__.items():
        clean_attribute(k, v)