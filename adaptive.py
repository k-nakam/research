import torch
from torch import nn
from torch.nn import functional as F

#Original code is https://github.com/facebookresearch/adaptive-softmax/tree/master/utils
#I also looked at the pytorch translation from https://github.com/rosinality/adaptive-softmax-pytorch
# and https://pytorch.org/docs/stable/_modules/torch/nn/modules/adaptive.html (highly cited)
#For adaptive inputs, I also looked at https://github.com/AranKomat/adapinp/blob/master/AdapInp.py

from collections import namedtuple
_ASMoutput = namedtuple('_ASMoutput', ['output', 'loss'])

class AdaptiveSoftmax(nn.Module):
    def __init__(self, ninp, ntokens, cutoffs, div_value = 4):
        super(AdaptiveSoftmax, self).__init__()

        self.ninp = ninp
        self.ntokens = ntokens
        self.cutoffs = cutoffs + [ntokens]

        self.head = nn.Linear(self.ninp, self.cutoffs[0] + len(self.cutoffs) - 1, bias=False)
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoffs) - 1):
            prj = nn.Sequential(
                nn.Linear(self.ninp, self.ninp // (div_value ** (i + 1)), bias=False),
                nn.Linear(self.ninp // (div_value ** (i + 1)), self.cutoffs[i + 1] - self.cutoffs[i], bias=False),
            )

            self.tail.append(prj)

    def forward(self, input, target):
        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            #making clusters: the first cluster would contain the most frequent words
            target_mask = (target >= cutoff_values[i]) & (target < cutoff_values[i+1])
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])
            else:
                relative_target = target[target_mask] - cutoff_values[i]
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.cutoffs[0] + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)

        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean() #NLL

        return _ASMoutput(output, loss)

class AdaptiveInput(nn.Module):
    def __init__(self, ninp, ntokens, cutoffs, div_value=4):
        super(AdaptiveInput, self).__init__()
        self.ninp = ninp
        self.ntokens = ntokens
        self.cutoffs = cutoffs + [ntokens]

        self.head = nn.Embedding(self.cutoffs[0], self.ninp)

        self.tail = nn.ModuleList()
        for i in range(len(self.cutoffs) - 1):
            prj = nn.Sequential(
                nn.Embedding(self.cutoffs[i + 1] - self.cutoffs[i],  ninp // (div_value ** (i + 1))),
                nn.Linear(ninp // (div_value ** (i + 1)), ninp, bias=False),
                nn.Dropout(0.5),
            )

            self.tail.append(prj)

    def forward(self, input):
        used_rows = 0
        input_size = list(input.size())
        output = input.new_zeros(input_size + [self.ninp]).float()

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            #making clusters
            input_mask = (input >= cutoff_values[i]) & (input < cutoff_values[i+1])
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            if i == 0:
                out = self.head(input[input_mask] - cutoff_values[i])
            else:
                out = self.tail[i - 1](input[input_mask] - cutoff_values[i])
            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        return output
