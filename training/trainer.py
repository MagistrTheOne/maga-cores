"""Training primitives with SGD written as the literal update equation."""
import torch

class TransparentSGD:
    """Classic SGD, optionally with momentum.

    AdamW became standard because per-parameter variance scaling converges reliably
    and decoupled weight decay regularizes cleanly. We use SGD so W=W-lr*grad stays
    visible. We trade away AdamW's adaptive scaling: convergence is slower and the
    learning rate is more sensitive to gradient/feature scale.
    """
    def __init__(self, parameters, learning_rate, momentum=0.0):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]

    @torch.no_grad()
    def step(self):
        """INPUT parameters and same-shaped gradients -> updated parameters."""
        for parameter, velocity in zip(self.parameters, self.velocities):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if self.momentum:
                velocity.mul_(self.momentum)
                velocity.add_(gradient)
                gradient = velocity
            parameter.add_(gradient, alpha=-self.learning_rate)  # W = W - lr * grad

    def zero_grad(self):
        """INPUT accumulated gradients -> OUTPUT gradients cleared to None."""
        for parameter in self.parameters:
            parameter.grad = None

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
