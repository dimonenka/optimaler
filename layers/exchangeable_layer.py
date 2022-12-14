import torch
from torch import nn


class Exchangeable(nn.Module):
    def __init__(self, in_channels, out_channels, row=True, col=True, mat=True, add_channel_dim=False, sum_layers=False):
        super().__init__()

        self.channel_layer = nn.Linear(in_channels, out_channels)

        self.row = row
        self.col = col
        self.mat = mat
        self.add_channel_dim = add_channel_dim
        self.sum_layers = sum_layers

        if self.row:
            self.row_layer = nn.Linear(in_channels, out_channels, bias=False)
            if sum_layers:
                self.row_layer_sum = nn.Linear(in_channels, out_channels, bias=False)
        if self.col:
            self.col_layer = nn.Linear(in_channels, out_channels, bias=False)
            if sum_layers:
                self.col_layer_sum = nn.Linear(in_channels, out_channels, bias=False)
        if self.mat:
            self.mat_layer = nn.Linear(in_channels, out_channels, bias=False)
            if sum_layers:
                self.mat_layer_sum = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        """
        :param x: [bs, rows, columns, channels] if not self.add_channel_dim else [bs, rows, columns]
        """

        if self.add_channel_dim:
            x = x.unsqueeze(-1)
        assert len(x.shape) == 4, "Exchangeable accepts tensors shaped as [bs, rows, columns, channels]"

        out = self.channel_layer(x)

        if self.row:
            row = torch.mean(x, dim=1, keepdim=True)
            out += self.row_layer(row)
            if self.sum_layers:
                row_sum = torch.sum(x, dim=1, keepdim=True)
                out += self.row_layer_sum(row_sum)

        if self.col:
            col = torch.mean(x, dim=2, keepdim=True)
            out += self.col_layer(col)
            if self.sum_layers:
                col_sum = torch.sum(x, dim=2, keepdim=True)
                out += self.col_layer_sum(col_sum)

        if self.mat:
            if self.row:
                mat = torch.mean(row, dim=2, keepdim=True)
                if self.sum_layers:
                    mat_sum = torch.sum(row_sum, dim=2, keepdim=True)
            elif self.col:
                mat = torch.mean(col, dim=1, keepdim=True)
                if self.sum_layers:
                    mat_sum = torch.sum(col_sum, dim=1, keepdim=True)
            else:
                mat = torch.mean(x, dim=[1, 2], keepdim=True)
                if self.sum_layers:
                    mat_sum = torch.sum(x, dim=[1, 2], keepdim=True)
            out += self.mat_layer(mat)
            if self.sum_layers:
                out += self.mat_layer_sum(mat_sum)

        return out
