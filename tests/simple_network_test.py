from torch import nn
from utils import model_parameter_size

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

model = NN()
total_params, trainable_params = model_parameter_size(model)
print('Total parameters:', total_params)
print('Trainable parameters:', trainable_params)
