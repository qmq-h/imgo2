import torch
import torch.nn as nn


class EmpiricalNormalizer(nn.Module):
    """Torch running mean/variance normalizer saved as part of a checkpoint."""

    def __init__(self, size, *, clip=10.0, epsilon=1.0e-4):
        super().__init__()
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, values):
        values = values.detach().reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.new_tensor(float(values.shape[0]))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m_2 / total)
        self.count.copy_(total)

    def forward(self, values, *, update=False):
        if update:
            self.update(values)
        normalized = (values - self.mean) / torch.sqrt(self.var + self.epsilon)
        return normalized.clamp(-self.clip, self.clip)

