import torch
import torch.optim as optim
from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from torch import nn

from .base_exploration_model import BaseExplorationModel


def init_method_1(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_().to(ptu.device)
            module.bias.data.uniform_().to(ptu.device)


def init_method_2(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_().to(ptu.device)
            module.bias.data.normal_().to(ptu.device)


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams["ob_dim"]
        self.output_size = hparams["rnd_output_size"]
        self.n_layers = hparams["rnd_n_layers"]
        self.size = hparams["rnd_size"]
        self.optimizer_spec = optimizer_spec

        # <TODO>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
        )
        # Randomly initialize weight
        init_method_1(self.f)
        self.f.to(ptu.device)

        # 2) f_hat, the function we are using to learn f
        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
        )
        # Randomly initialize a different weight
        init_method_2(self.f_hat)
        self.f_hat.to(ptu.device)

        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(), **self.optimizer_spec.optim_kwargs
        )

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

    def forward(self, ob_no):
        # <TODO>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        if not isinstance(ob_no, torch.Tensor):
            ob_no = ptu.from_numpy(ob_no.copy())

        return (self.f_hat(ob_no) - self.f(ob_no).detach()).abs().mean(axis=-1)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <TODO>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        if not isinstance(ob_no, torch.Tensor):
            ob_no = ptu.from_numpy(ob_no.copy())

        error = self(ob_no)
        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return loss.item()
