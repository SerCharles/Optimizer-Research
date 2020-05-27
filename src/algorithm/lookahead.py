from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):


    def __init__(self, optimizer, k=5, inner_lr=0.8):
        '''
        描述：初始化Lookahead优化器
        参数：主优化器，slow_rates_step经过k次才更新，slow_rates_step更新的lr
        返回：无
        '''
        self.optimizer = optimizer
        self.k = k
        self.inner_lr = inner_lr
        self.current_k = 0
        self.state = defaultdict(dict)
        
        #用cached params来存储
        for group in optimizer.param_groups:
            for p in group['params']:
                self.state[p]['slow_weights'] = torch.zeros_like(p.data)
                self.state[p]['slow_weights'].copy_(p.data)
    



    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        #更新fast rate，此时得到的p.data是fast rate
        loss = self.optimizer.step(closure)
        self.current_k += 1
        #更新slow rate
        if(self.current_k >= self.k):
            self.current_k = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    #slow = alpha*(fast - slow) + slow
                    #slow = alpha*fast + (1 - alpha)*slow
                    fast_weights = p.data
                    slow_weights = self.state[p]['slow_weights']
                    slow_weights.mul_(1 - self.inner_lr).add_(self.inner_lr, fast_weights)
                    p.data.copy_(slow_weights)
        return loss

