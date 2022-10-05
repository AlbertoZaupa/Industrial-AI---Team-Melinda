# torch.nn.RNN(input_size, hidden_layer, num_layer, bias=True, batch_first=False, dropout = 0, bidirectional = False)
#
# Parameters:
#
# input_size: In input x the number of expected features. hidden_layer: The number of features in the hidden state.
# num_layer: The num_layer is used as several recurrent layers. bias: If the bias is False then the layer does not
# use bias weights. batch_first: If batch_first is True then input and output tensors are provided ( batch, seq,
# feature) instead of (seq, batch, feature). The default value of batch_first is False. dropout: If non-zero,
# initiate the dropout layer on the output of each RNN layer excluding the last layer with a dropout probability
# equal to dropout. The default value of dropout is 0. bidirectional: If True, then it becomes a bidirectional RNN.
# The default value of bidirectional is False.

# torch.nn.RNNCell(input_size, hidden_size, bias = True, nonlinearity = 'tanh', device = None, dtype = None)
#
# Parameters:
#
#     input_size the number of expected features in the input x.
#     hidden_size the number of features in the hidden state as h.
#     bias If the bias is False then the layer does not use bias weight. The default value of bias is True.
#     nonlinearity The default nonlinearity is tanh. It used can be either tanh or relu.


import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.datasets as dtsets

torchdata.

traindt = dtsets.MNIST(root='./data',
                       train=True,
                       transform=transform.ToTensor(),
                       download=True)

testdt = dtsets.MNIST(root='./data',
                      train=False,
                      transform=transform.ToTensor())
batchsiz = 80
nitrs = 2800
numepoch = nitrs / (len(traindt) / batchsiz)
numepoch = int(numepoch)

trainldr = torch.utils.data.DataLoader(dataset=traindt,
                                       batch_size=batchsiz,
                                       shuffle=True)

testldr = torch.utils.data.DataLoader(dataset=testdt,
                                      batch_size=batchsiz,
                                      shuffle=False)


class rnn(nn.Module):
    def __init__(self, inpdim, hidendim, layerdim, outpdim):
        super(rnn, self).__init__()
        self.hidendim = hidendim

        self.layerdim = layerdim

        self.rnn = nn.RNN(inpdim, hidendim, layerdim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidendim, outpdim)

    def forward(self, y):
        h = torch.zeros(self.layerdim, y.size(0), self.hidendim).requires_grad_()
        outp, hx = self.rnn(y, h.detach())
        outp = self.fc(outp[:, -1, :])
        return outp


inpdim = 28
hidendim = 80
layerdim = 1
outpdim = 10
mdl = rnn(inpdim, hidendim, layerdim, outpdim)
criter = nn.CrossEntropyLoss()
l_r = 0.01

optim = torch.optim.SGD(mdl.parameters(), lr=l_r)
list(mdl.parameters())[0].size()
seqdim = 28

itr = 0
for epoch in range(numepoch):
    for x, (imgs, lbls) in enumerate(trainldr):
        mdl.train()
        imgs = imgs.view(-1, seqdim, inpdim).requires_grad_()
        optim.zero_grad()
        outps = mdl(imgs)
        loss = criter(outps, lbls)
        loss.backward()

        optim.step()

        itr += 1

        if itr % 500 == 0:
            mdl.eval()
            crrct = 0
            ttl = 0
            for imgs, lbls in testldr:
                imgs = imgs.view(-1, seqdim, inpdim)

                outps = mdl(imgs)
                _, predicted = torch.max(outps.data, 1)

                ttl += lbls.size(0)

                crrct += (predicted == lbls).sum()

            accu = 100 * crrct / ttl

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accu))
