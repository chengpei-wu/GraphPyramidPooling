import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="avg"):
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x, num_per_batch, degrees):
        return self.temporal_pyramid_pool(x, self.levels, self.mode, num_per_batch, degrees)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode, num_per_batch, degrees):
        num_per_batch = num_per_batch.tolist()
        degrees = list(enumerate(degrees.tolist()))
        cut_index = 0
        all_degree = []
        for i in num_per_batch:
            all_degree.append(degrees[cut_index:cut_index + i])
            cut_index += i
        ranked_all_degree = []
        for d in all_degree:
            t = sorted(d, key=lambda x: x[1], reverse=True)
            tt = [item[0] for item in t]
            for i in tt:
                ranked_all_degree.append(i)
        ranked_all_degree = torch.tensor(ranked_all_degree)
        previous_conv = previous_conv[ranked_all_degree, :]
        # print(previous_conv.shape)

        cut_index = 0
        for kk, k in enumerate(num_per_batch):
            feats = torch.t(previous_conv[cut_index:cut_index + k, :])
            cut_index += k
            num_sample = feats.size(0)
            len_tensor = feats.size(1)
            for i, pool_size in zip(range(len(out_pool_size)), out_pool_size):
                kernel = int(math.ceil(len_tensor / pool_size))
                pad1 = int(math.floor((kernel * pool_size - len_tensor) / 2))
                pad2 = int(math.ceil((kernel * pool_size - len_tensor) / 2))
                assert pad1 + pad2 == (kernel * pool_size - len_tensor)

                padded_input = F.pad(input=feats, pad=[0, pad1 + pad2],
                                     mode='constant', value=0)
                if mode == "max":
                    pool = nn.MaxPool1d(kernel_size=kernel, stride=kernel, padding=0)
                elif mode == "avg":
                    pool = nn.AvgPool1d(kernel_size=kernel, stride=kernel, padding=0)
                else:
                    raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
                x = pool(padded_input)
                if i == 0:
                    tpp = x.view(num_sample, -1)
                else:
                    tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)
            print(tpp.shape)
            if kk == 0:
                read_out = tpp.view(tpp.size(0) * tpp.size(1), -1)
            else:
                read_out = torch.cat((read_out, tpp.view(tpp.size(0) * tpp.size(1), -1)), 1)
        return torch.t(read_out)
