import torch
from src.modeling_llama_if import _make_causal_mask


def _make_causal_mask(length, dtype):
    tgt_len = length
    mask = torch.full((tgt_len, tgt_len), 0)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), torch.finfo(dtype).min)
    return mask.to(dtype)


def make_tree_attention(num_candidate, tree_type, dtype, device, mask_num, return_index=False):

    activate_idx = []

    if tree_type == "full":

        if mask_num == 3:
            if num_candidate == 2:
                mask = _make_causal_mask(14, dtype)
                mask.masked_fill_(torch.eye(14).bool(), 0)

                mask[2, 0] = 0
                mask[3, 0] = 0
                mask[4, 1] = 0
                mask[5, 1] = 0

                mask[6, 0], mask[6, 2] = 0, 0
                mask[7, 0], mask[7, 2] = 0, 0
                mask[8, 0], mask[8, 3] = 0, 0
                mask[9, 0], mask[9, 3] = 0, 0
                mask[10, 1], mask[10, 4] = 0, 0
                mask[11, 1], mask[11, 4] = 0, 0
                mask[12, 1], mask[12, 5] = 0, 0
                mask[13, 1], mask[13, 5] = 0, 0

            elif num_candidate == 3:
                mask = _make_causal_mask(39, dtype)
                mask.masked_fill_(torch.eye(39).bool(), 0)

                mask[3, 0] = 0
                mask[4, 0] = 0
                mask[5, 0] = 0
                mask[6, 1] = 0
                mask[7, 1] = 0
                mask[8, 1] = 0
                mask[9, 2] = 0
                mask[10, 2] = 0
                mask[11, 2] = 0

                mask[12, 0], mask[12, 3] = 0, 0
                mask[13, 0], mask[13, 3] = 0, 0
                mask[14, 0], mask[14, 3] = 0, 0
                mask[15, 0], mask[15, 4] = 0, 0
                mask[16, 0], mask[16, 4] = 0, 0
                mask[17, 0], mask[17, 4] = 0, 0
                mask[18, 0], mask[18, 5] = 0, 0
                mask[19, 0], mask[19, 5] = 0, 0
                mask[20, 0], mask[20, 5] = 0, 0
                mask[21, 1], mask[21, 6] = 0, 0
                mask[22, 1], mask[22, 6] = 0, 0
                mask[23, 1], mask[23, 6] = 0, 0
                mask[24, 1], mask[24, 7] = 0, 0
                mask[25, 1], mask[25, 7] = 0, 0
                mask[26, 1], mask[26, 7] = 0, 0
                mask[27, 1], mask[27, 8] = 0, 0
                mask[28, 1], mask[28, 8] = 0, 0
                mask[29, 1], mask[29, 8] = 0, 0
                mask[30, 2], mask[30, 9] = 0, 0
                mask[31, 2], mask[31, 9] = 0, 0
                mask[32, 2], mask[32, 9] = 0, 0
                mask[33, 2], mask[33, 10] = 0, 0
                mask[34, 2], mask[34, 10] = 0, 0
                mask[35, 2], mask[35, 10] = 0, 0
                mask[36, 2], mask[36, 11] = 0, 0
                mask[37, 2], mask[37, 11] = 0, 0
                mask[38, 2], mask[38, 11] = 0, 0

            elif num_candidate == 4:
                mask = _make_causal_mask(84, dtype)
                mask.masked_fill_(torch.eye(84).bool(), 0)

                mask[4, 0] = 0
                mask[5, 0] = 0
                mask[6, 0] = 0
                mask[7, 0] = 0
                mask[8, 1] = 0
                mask[9, 1] = 0
                mask[10, 1] = 0
                mask[11, 1] = 0
                mask[12, 2] = 0
                mask[13, 2] = 0
                mask[14, 2] = 0
                mask[15, 2] = 0
                mask[16, 3] = 0
                mask[17, 3] = 0
                mask[18, 3] = 0
                mask[19, 3] = 0

                mask[20, 0], mask[20, 4] = 0, 0
                mask[21, 0], mask[21, 4] = 0, 0
                mask[22, 0], mask[22, 4] = 0, 0
                mask[23, 0], mask[23, 4] = 0, 0
                mask[24, 0], mask[24, 5] = 0, 0
                mask[25, 0], mask[25, 5] = 0, 0
                mask[26, 0], mask[26, 5] = 0, 0
                mask[27, 0], mask[27, 5] = 0, 0
                mask[28, 0], mask[28, 6] = 0, 0
                mask[29, 0], mask[29, 6] = 0, 0
                mask[30, 0], mask[30, 6] = 0, 0
                mask[31, 0], mask[31, 6] = 0, 0
                mask[32, 0], mask[32, 7] = 0, 0
                mask[33, 0], mask[33, 7] = 0, 0
                mask[34, 0], mask[34, 7] = 0, 0
                mask[35, 0], mask[35, 7] = 0, 0
                mask[36, 1], mask[36, 8] = 0, 0
                mask[37, 1], mask[37, 8] = 0, 0
                mask[38, 1], mask[38, 8] = 0, 0
                mask[39, 1], mask[39, 8] = 0, 0
                mask[40, 1], mask[40, 9] = 0, 0
                mask[41, 1], mask[41, 9] = 0, 0
                mask[42, 1], mask[42, 9] = 0, 0
                mask[43, 1], mask[43, 9] = 0, 0
                mask[44, 1], mask[44, 10] = 0, 0
                mask[45, 1], mask[45, 10] = 0, 0
                mask[46, 1], mask[46, 10] = 0, 0
                mask[47, 1], mask[47, 10] = 0, 0
                mask[48, 1], mask[48, 11] = 0, 0
                mask[49, 1], mask[49, 11] = 0, 0
                mask[50, 1], mask[50, 11] = 0, 0
                mask[51, 1], mask[51, 11] = 0, 0
                mask[52, 2], mask[52, 12] = 0, 0
                mask[53, 2], mask[53, 12] = 0, 0
                mask[54, 2], mask[54, 12] = 0, 0
                mask[55, 2], mask[55, 12] = 0, 0
                mask[56, 2], mask[56, 13] = 0, 0
                mask[57, 2], mask[57, 13] = 0, 0
                mask[58, 2], mask[58, 13] = 0, 0
                mask[59, 2], mask[59, 13] = 0, 0
                mask[60, 2], mask[60, 14] = 0, 0
                mask[61, 2], mask[61, 14] = 0, 0
                mask[62, 2], mask[62, 14] = 0, 0
                mask[63, 2], mask[63, 14] = 0, 0
                mask[64, 2], mask[64, 15] = 0, 0
                mask[65, 2], mask[65, 15] = 0, 0
                mask[66, 2], mask[66, 15] = 0, 0
                mask[67, 2], mask[67, 15] = 0, 0
                mask[68, 3], mask[68, 16] = 0, 0
                mask[69, 3], mask[69, 16] = 0, 0
                mask[70, 3], mask[70, 16] = 0, 0
                mask[71, 3], mask[71, 16] = 0, 0
                mask[72, 3], mask[72, 17] = 0, 0
                mask[73, 3], mask[73, 17] = 0, 0
                mask[74, 3], mask[74, 17] = 0, 0
                mask[75, 3], mask[75, 17] = 0, 0
                mask[76, 3], mask[76, 18] = 0, 0
                mask[77, 3], mask[77, 18] = 0, 0
                mask[78, 3], mask[78, 18] = 0, 0
                mask[79, 3], mask[79, 18] = 0, 0
                mask[80, 3], mask[80, 19] = 0, 0
                mask[81, 3], mask[81, 19] = 0, 0
                mask[82, 3], mask[82, 19] = 0, 0
                mask[83, 3], mask[83, 19] = 0, 0
            else:
                raise ValueError
        else:
            raise ValueError

    elif tree_type == "half":
        mask = _make_causal_mask(mask_num * num_candidate, dtype)
        mask.masked_fill_(torch.eye(mask_num * num_candidate).bool(), 0)

        for m in range(1, mask_num):
            mask[m * num_candidate:, (m - 1) * num_candidate] = 0
    
    elif tree_type == "upper-triangle":
        num_c = 0
        num_candidates = []
        num_accumulate = [0]
        for m in range(mask_num):
            num_c += num_candidate - m
            num_candidates.append(num_candidate - m)
            num_accumulate.append(num_c)
        mask = _make_causal_mask(num_c, dtype)
        mask.masked_fill_(torch.eye(num_c).bool(), 0)

        for m in range(1, mask_num):
            mask[num_accumulate[m]:, num_accumulate[m - 1]] = 0

    if return_index:
        for m in range(mask.shape[0]):
            _mask = mask[m][:(m + 1)]
            _idx = torch.nonzero(_mask == 0).view(-1).tolist()
            activate_idx.append(_idx)
        return mask.to(device), activate_idx
    else:            
        return mask.to(device)