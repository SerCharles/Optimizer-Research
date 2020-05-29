import torch
import math
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
		length_SMA_max = 2 / (1 - betas[1]) - 1
		defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "length_SMA_max": length_SMA_max}
		super(RAdam, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(RAdam, self).__setstate__(state)

	def step(self, closure=None):
		loss = None
		if closure:
			loss = closure()

		for param_group in self.param_groups:
			for param in param_group["params"]:
				if param.grad is None:
					continue
				if param.is_sparse:
					raise RuntimeError("RAdam does not support sparse gradients")
				gra = param.grad.data.float()
				data = param.data.float()

				state = self.state[param]
				if not state:
					state["step"] = 0
					state["v"] = torch.zeros_like(gra)
					state["m"] = torch.zeros_like(gra)
				v = state["v"]
				m = state["m"]
				state["step"] += 1
				beta1, beta2 = param_group["betas"]

				v.mul_(beta2).addcmul_(1-beta2, gra, gra)
				m.mul_(beta1).add_(1-beta1, gra)
				m = m / (1 - beta1**state["step"])
				length_SMA = param_group["length_SMA_max"] - 2 * state["step"] * (beta2 ** state["step"] / (1 - beta2 ** state["step"]))

				if length_SMA > 5:
					if param_group["weight_decay"] != 0:
						data.add_(-param_group["weight_decay"] * param_group["lr"], data)
					rectifier = param_group["lr"] * math.sqrt((length_SMA - 4) * (length_SMA - 2) * param_group["length_SMA_max"] / (param_group["length_SMA_max"] - 4) / (param_group["length_SMA_max"] - 2) / length_SMA * (1 - beta2 ** state["step"]))
					std = v.sqrt().add_(param_group["eps"])
					data.addcdiv_(-rectifier, m, std)
				else:
					if param_group["weight_decay"] != 0:
						data.add_(-param_group["weight_decay"] * param_group["lr"], data)
					data.add_(-param_group["lr"], m)
		return loss
