import torch
import torch.nn as nn
from torch.autograd import Variable

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(
            torch.zeros(input.size(0), 51), requires_grad=False)
        c_t = Variable(
            torch.zeros(input.size(0), 51), requires_grad=False)
        h_t2 = Variable(
            torch.zeros(input.size(0), 1), requires_grad=False)
        c_t2 = Variable(
            torch.zeros(input.size(0), 1), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]

        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(c_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
