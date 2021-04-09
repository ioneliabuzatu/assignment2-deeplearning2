import torch
import torch.nn as nn


class HighwayLayer(nn.Module):
    """ Fully connected layer in a highway network. """

    def __init__(self, in_features: int, out_features: int, carry: bool = False, act: callable = torch.tanh):
        """
        Parameters
        ----------
        in_features : int
            Number of input dimensions.
        out_features : int
            Number of output dimensions.
        carry : bool
            Whether or not to use a separate carry gate.
        act : callable
            Activation function to use.
        """
        super(HighwayLayer, self).__init__()
        self.non_square = False
        self.linear_layer = nn.Linear(in_features, out_features, bias=True)
        self.activation = act
        self.carry = carry
        if in_features != out_features:
            self.non_square = True

    def forward(self, input):
        t = self.activation(input)
        f = self.linear_layer(input)

        if not self.carry:
            if self.non_square:
                input_gating = self.linear_layer(input).T * ((1-t)*input)
            else:
                input_gating = (1 - t) * input

            return input_gating.T + (t.T * f)

        if self.non_square:
            input_gating = self.linear_layer(input).T * (t*input)
        else:
            input_gating = t * input

        return input_gating.T + t.T*f


highway = HighwayLayer(100, 100)
highway(torch.randn(1, 100))
print(sum([par.numel() for par in highway.parameters()]))

highway = HighwayLayer(1000, 10)
highway(torch.randn(1, 1000))
print(sum([par.numel() for par in highway.parameters()]))

highway = HighwayLayer(100, 100, carry=True)
highway(torch.randn(1, 100))
print(sum([par.numel() for par in highway.parameters()]))

highway = HighwayLayer(1000, 10, carry=True)
highway(torch.randn(1, 1000))
print(sum([par.numel() for par in highway.parameters()]))
