from loss import InfoNCE
from transformers import Trainer
import torch
import torch.nn as nn

def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        temperature = kwargs.get("args").temperature
        self.info_nce = InfoNCE(temperature=temperature,
                                device=self.accelerator.device)
        self.grads = None
        self.alpha = kwargs.get("args").alpha
        self.lamb = kwargs.get("args").lamb
    
    def encode(self, model, x):
        out = model(**x, output_hidden_states=True).hidden_states[-1][:, -1, :]
        return out
    
    def compute_loss(self, model, inputs, return_outputs=False):
        sent0 = {'input_ids': inputs.get('sent0_input_ids'),
                'attention_mask': inputs.get('sent0_attention_mask')}
        sent1 = {'input_ids': inputs.get('sent1_input_ids'),
                'attention_mask': inputs.get('sent1_attention_mask')}
        hard_neg = {'input_ids': inputs.get('hard_neg_input_ids'),
                    'attention_mask': inputs.get('hard_neg_attention_mask')}
        
        sent0_embed = self.encode(model, sent0)
        sent1_embed = self.encode(model, sent1)
        hard_neg_embed = self.encode(model, hard_neg)
        loss = self.info_nce(sent0_embed, sent1_embed, hard_neg_embed)
        return loss

    def training_step(self, model, inputs):
        loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        
        # Apply gradient filtering with EMA
        self.grads = gradfilter_ema(model, grads=self.grads, alpha=self.alpha, lamb=self.lamb)
        
        self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.detach()
