from typing import Dict, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# basic MLP class (used for different ICM networks)
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Sequence[int] = (256, 256)):
        # print("MLP in_dim:", input_dim, type(input_dim))
        # print("MLP hidden:", hidden, [type(h) for h in hidden])
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ICMResults:
    def __init__(self, r_int: torch.Tensor, loss: torch.Tensor, info: Dict[str, torch.Tensor]):
        self.r_int = r_int
        self.loss = loss
        self.info = info
# this is based on Pathak et al's implementation of ICM
class ICM(nn.Module):
    def __init__(
        self,
        obs_dim: int = 60,
        action_dim: int = 7,
        feature_dim: int = 128,
        beta: float = 0.2,
        forward_scale: float = 0.5,
        hidden = (256, 256),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.beta = beta
        self.forward_scale = forward_scale
        
        # this is the first neural net: encoder
        self.encoder = MLP(obs_dim, feature_dim, hidden)

        # this is the second neural net: inverse model
        self.inverse_model = MLP(2 * feature_dim, action_dim, hidden)

        # this is the third neural net: forward model
        self.forward_model = MLP(feature_dim + action_dim, feature_dim, hidden)
    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor) -> ICMResults:
        # cast to floats
        obs = obs.float()
        next_obs = next_obs.float()
        action = action.float()

        # encode the observations
        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)

        # predict the action using the inverse model
        inverse_input = torch.cat([phi, phi_next], dim=1)
        pred_action = self.inverse_model(inverse_input)
        inverse_loss = F.mse_loss(pred_action, action)


        # predict the next state feature using the forward model
        forward_input = torch.cat([phi, action], dim=1)
        predict_next_phi = self.forward_model(forward_input)

        # compute prediction error and loss to use to compute intrinsic reward
        forward_error = F.mse_loss(predict_next_phi, phi_next.detach(), reduction='none').mean(dim=1)
        forward_loss = forward_error.mean()

        # intrinsic reward
        r_int = (self.forward_scale * forward_error).detach() # detach so no backprop

        loss = 1 - self.beta * inverse_loss + self.beta * forward_loss

        # to show on tensorboard..
        info = {
            "icm_loss": loss.detach(),
            "inv_loss": inverse_loss.detach(),
            "fwd_loss": forward_loss.detach(),
            "r_int_mean": r_int.mean().detach(),
            "r_int_std": r_int.std(unbiased=False).detach(),
        }

        return ICMResults(r_int, loss, info)
        
     