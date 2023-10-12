from torch import nn
from torch.nn import functional as F

class ModelNeo(nn.Module):
    def __init__(self):
        super(ModelNeo, self).__init__()
        asam = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
        self.embedding = nn.Embedding(len(asam), 21)

        self.lstm1 = nn.LSTM(21, 64, 1)
        self.lstm2 = nn.LSTM(64, 64, 1)
        self.lstm3 = nn.LSTM(64, 64, 1)
        self.lstm4 = nn.LSTM(64, 64, 1)
        self.batchnormL1 = nn.BatchNorm1d(64)
        self.batchnormL2 = nn.BatchNorm1d(64)
        self.batchnormL3 = nn.BatchNorm1d(64)
        self.batchnormL4 = nn.BatchNorm1d(64)
        self.dropL1 = nn.Dropout(p=0.5)
        self.dropL2 = nn.Dropout(p=0.5)
        self.dropL3 = nn.Dropout(p=0.5)
        self.dropL4 = nn.Dropout(p=0.5)
        self.linearL1 = nn.Linear(64, 32)

        self.conv1 = nn.Conv2d(1, 64, (3,21), padding = (1,0))
        self.conv2 = nn.Conv2d(64, 64, (3,1), padding = (1,0))
        self.conv3 = nn.Conv2d(64, 64, (3,1), padding = (1,0))
        self.conv4 = nn.Conv2d(64, 64, (3,1), padding = (1,0))
        self.batchnormC1 = nn.BatchNorm2d(64)
        self.batchnormC2 = nn.BatchNorm2d(64)
        self.batchnormC3 = nn.BatchNorm2d(64)
        self.batchnormC4 = nn.BatchNorm2d(64)
        self.dropC1 = nn.Dropout(p=0.5)
        self.dropC2 = nn.Dropout(p=0.5)
        self.dropC3 = nn.Dropout(p=0.5)
        self.dropC4 = nn.Dropout(p=0.5)
        #self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linearC1 = nn.Linear(1216, 32)

        self.linear2 = nn.Linear(32, 2)

    def forward(self, x):
        emb = self.embedding(x)

        #pdb.set_trace()
        lstm = emb.permute(1, 0, 2)

        lstm = self.lstm1(lstm)
        # lstm = lstm[0].permute(1,2,0)
        # lstm = self.batchnormL1(lstm)
        # lstm = lstm.permute(2,0,1)
        lstm1 = self.dropL1(lstm[0])

        lstm = self.lstm2(lstm1)
        # lstm = lstm[0].permute(1,2,0)
        # lstm = self.batchnormL2(lstm)
        # lstm = lstm.permute(2,0,1)
        lstm = self.dropL2(lstm[0])
        lstm2 = lstm + lstm1

        # lstm = self.lstm3(lstm2)
        # lstm = lstm[0].permute(1,2,0)
        # lstm = self.batchnormL3(lstm)
        # lstm = lstm.permute(2,0,1)
        # lstm = self.dropL3(lstm[0])
        # lstm3 = lstm + lstm2

        # lstm = self.lstm4(lstm3)
        # lstm = lstm[0].permute(1,2,0)
        # lstm = self.batchnormL4(lstm)
        # lstm = lstm.permute(2,0,1)
        # lstm = self.dropL4(lstm)
        # lstm4 = lstm + lstm3
        # lstm = lstm2

        lstm = self.linearL1(lstm2[lstm2.shape[0]-1])

        # channel, add dim, seq, emd vec
        cnn = emb.view(emb.shape[0], 1, emb.shape[1], emb.shape[2])

        cnn = self.conv1(cnn)
        cnn = F.relu(cnn)
        # cnn = self.batchnormC1(cnn)
        cnn1 = self.dropC1(cnn)

        cnn = self.conv2(cnn1)
        cnn = F.relu(cnn)
        # cnn = self.batchnormC2(cnn)
        cnn = self.dropC2(cnn)
        cnn2 = cnn + cnn1

        cnn = self.conv3(cnn2)
        cnn = F.relu(cnn)
        # cnn = self.batchnormC3(cnn)
        cnn = self.dropC3(cnn)
        cnn3 = cnn + cnn2

        cnn = self.conv4(cnn3)
        cnn = F.relu(cnn)
        # cnn = self.batchnormC4(cnn)
        cnn = self.dropC4(cnn)
        cnn4 = cnn + cnn3

        #cnn = self.pool(cnn)
        cnn = self.flatten(cnn4)
        cnn = self.linearC1(cnn)

        out = cnn+lstm
        y_pred = self.linear2(out)
        return y_pred
