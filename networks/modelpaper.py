import torch.nn
from torch import nn


class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.body = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, batch):
        x = batch.reshape((-1, self.input_size))
        body_out = self.body(x)
        return body_out

    def clone(self):
        new_state_dict = {}
        for kw, v in self.state_dict().items():
            new_state_dict[kw] = v.clone()
        new_net = Net(self.input_size)
        new_net.load_state_dict(new_state_dict)
        return new_net