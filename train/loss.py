import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class InfoNCE(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, query, pos, neg):
        '''
        Use other samples in batch as negative samples.
        query, pos, neg : [B, E]
        where B is a batch_size, E is an embedding size
        '''
        # Normalize
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
        
        # Compute cosine similarity locally
        logits_pos = query @ pos.T
        logits_neg = query @ neg.T
        
        # Concatenate logits
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        
        # Generate labels
        local_batch_size = query.shape[0]
        labels = torch.arange(local_batch_size).to(self.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        return loss
