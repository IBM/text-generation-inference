import torch

# Shuffle columns of scales
def shuffle_and_replace_scales(state_dict, scales_name, col_perm):
    scales = state_dict[scales_name]
    assert len(col_perm) == scales.shape[1]
    
    shuffled_scales = scales[:,col_perm]
    state_dict[scales_name] = shuffled_scales

def unpack_shuffle_repack_and_replace_qzeros(state_dict, bits, qzeros_name, col_perm):
    qzeros = state_dict[qzeros_name]
    mask = 2**bits - 1
    pack_size = 32 // bits
    assert len(col_perm) == qzeros.shape[1] * pack_size
    
    #unpack
    unpacked_qzeros = torch.zeros((qzeros.shape[0], qzeros.shape[1]*pack_size), dtype=torch.int)
    for i in range(pack_size):
        unpacked_qzeros[:, i::pack_size] = (qzeros >> (i*bits)) & (mask) 

    # shuffle
    shuffled_qzeros = unpacked_qzeros[:,col_perm]

    # repack
    packed_qzeros = torch.zeros_like(qzeros)
    for i in range(pack_size):
        packed_qzeros |= (shuffled_qzeros[:, i::pack_size] & mask) << (i*bits)     
    
    state_dict[qzeros_name] = packed_qzeros
    
def shuffle_and_replace_qweight(state_dict, bits, group_size, qweight_name, g_idx_name=None, next_g_idx_name=None, stable=False):
    qweight = state_dict[qweight_name]

    # unpack qweight
    mask = 2**bits - 1
    pack_size = 32 // bits
    unpacked_qweight = torch.zeros((qweight.shape[0]*pack_size, qweight.shape[1]), dtype=torch.int)
    for i in range(pack_size):
        unpacked_qweight[i::pack_size] = (qweight >> (i*bits)) & (mask) 

    # reorder rows conditionally
    if not (g_idx_name is None):
        g_idx = state_dict[g_idx_name]
        row_perm = torch.argsort(g_idx, stable=stable)
        unpacked_qweight = unpacked_qweight[row_perm]
    
    # reorder columns conditionally
    if not (next_g_idx_name is None):
        next_g_idx = state_dict[next_g_idx_name]
        col_perm = torch.argsort(next_g_idx, stable=stable)
        unpacked_qweight = unpacked_qweight[:,col_perm]

    # pack qweight
    packed_qweight = torch.zeros_like(qweight)
    for i in range(pack_size):
        packed_qweight |= (unpacked_qweight[i::pack_size] & mask) << (i*bits) 

    # replace qweight with new reordered one in state_dict
    print(f'replacing {qweight_name}')
    state_dict[qweight_name] = packed_qweight
    
    if not (g_idx_name is None):
        print(f'replacing {g_idx_name}')
        state_dict[g_idx_name] = torch.arange(0, len(g_idx), dtype=torch.int) // group_size 