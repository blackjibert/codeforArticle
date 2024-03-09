from torch import nn
import torch.nn.functional as F
class Lstm(nn.Module):

    def __init__(self) -> None:
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(32 * 3, 128, batch_first=True, num_layers=5)
        self.line1 = nn.Linear(128, 128)
        self.line2 = nn.Linear(128, 10)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.line1(out[:, -1, :])
        out = self.line2(out)
        output = F.log_softmax(out, dim=1)
        return output