import torch
from icm import ICM

device = "cpu"

icm = ICM(obs_dim=60, action_dim=7, feature_dim=128).to(device)
opt = torch.optim.Adam(icm.parameters(), lr=1e-3)

B = 256
obs = torch.randn(B, 60, device=device)
next_obs = torch.randn(B, 60, device=device)
action = torch.randn(B, 7, device=device).clamp(-1, 1)

out = icm.forward(obs, next_obs, action)

print("r_int shape:", out.r_int.shape)     # should be [256]
print("loss:", float(out.loss))
print("info keys:", out.info.keys())

opt.zero_grad()
out.loss.backward()
opt.step()

print("backward/step OK")