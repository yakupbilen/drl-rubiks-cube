import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.resBlocks = nn.ModuleList()

        self.firstBlock = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
        )

        for _ in range(3):
            self.resBlocks.append(ResidualBlock(1000))

        self.finalLayer = nn.Linear(1000, 1)

    def forward(self, batch):
        x = batch.reshape((-1, self.input_size))
        out = self.firstBlock(x)

        for block in self.resBlocks:
            out = block(out)

        out = self.finalLayer(out)

        return out.squeeze()

    def clone(self):
        new_state_dict = {}
        for kw, v in self.state_dict().items():
            new_state_dict[kw] = v.clone()
        new_net = Net(self.input_size)
        new_net.load_state_dict(new_state_dict)
        return new_net


class ResidualBlock(nn.Module):
    def __init__(self, channelsIn):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(channelsIn, channelsIn),
            nn.ReLU(),
            nn.BatchNorm1d(channelsIn),
            nn.Linear(channelsIn, channelsIn),
        )

        self.combinedLayers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channelsIn),
        )

    def forward(self, states):
        return self.combinedLayers(self.layers(states) + states)
