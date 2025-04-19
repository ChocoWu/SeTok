
import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Union, Dict, List,Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict, is_dataclass
from transformers.utils import ModelOutput
from .utils import *
from .tokenizer import SetokTokenizer
from .detokenizer import SetokDeTokenizer
from ..loss import GANLoss, MultilabelContrastiveLoss
from ..utils import instantiate_from_config


@dataclass
class SetokOutput(ModelOutput):
    token_emb: torch.FloatTensor = None
    predict_emb: torch.FloatTensor = None
    loss: torch.FloatTensor = None,
    loss_log: Dict = None,



class SeTok(nn.Module):
    def __init__(self, 
                 tokenizer_config,
                 detokenizer_config,
                 rec_loss_config=None, 
                 contrastive_loss_config=None,
                 is_training=False,
                 **kwargs) -> None:
        super(SeTok).__init__()

        self.tokenizer_config = tokenizer_config
        self.detokenizer_config = detokenizer_config

        self.tokenizer = SetokTokenizer(**asdict(tokenizer_config))

        self.detokenizer = SetokDeTokenizer(**asdict(detokenizer_config))
        
        self.is_training = is_training
        if is_training:
            self.rec_loss = GANLoss(**asdict(rec_loss_config))
            self.contrastive_loss = MultilabelContrastiveLoss(**asdict(contrastive_loss_config))

    def get_tokenizer_config(self):
        return self.tokenizer_config
    
    def get_detokenizer_config(self):
        return self.detokenizer_config
    
    def get_tokenizer(self):
        return self.tokenizer

    def get_detokenizer(self):
        return self.detokenizer

    def tokenize(self, x):
        return self.tokenizer(x)

    def detokenize(self, x):
        return self.detokenizer(x)
    
    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    

    def compute_rec_loss(self, prediction, target, current_step):
        loss, log_dict = self.rec_loss(target, prediction, current_step)
        return loss, log_dict
    
    def compute_contrastive_loss(self, prediction, text):
        loss, log_dict = self.contrastive_loss(prediction, text)
        return loss, log_dict

    
    def forward(self, x, gold_image=None, text=None, return_dict=True, current_step=0):
        e_tokens, _, _ = self.tokenize(x)
        prediction = self.detokenize(e_tokens)
        loss = None
        loss_log = dict()
        if gold_image is not None:
            rec_loss, rec_loss_log = self.compute_rec_loss(prediction, gold_image, current_step)
            loss = rec_loss
            loss_log.update(**rec_loss_log)
        
        if text is not None:
            txt_contrastive_loss, txt_contrastive_log = self.compute_contrastive_loss(e_tokens, text)
            loss += txt_contrastive_loss
            loss_log.update(**txt_contrastive_log)
        if return_dict:
            SetokOutput(token_emb=e_tokens, predict_emb=prediction, loss=loss, loss_log=loss_log)
        else:
            loss, (e_tokens, prediction, loss_log)



