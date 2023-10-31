from torch import nn
class ProtBERT(nn.Module):

    def __init__(self, bert):
        super(ProtBERT, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(1024, 256)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(256, 2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):

        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
#         cls_hs = cls_hs[0].permute(0, 2, 1)

        output = self.fc1(cls_hs)

        output = self.relu(output)

        output = self.dropout(output)

        # output layer
        output = self.fc2(output)

        # apply softmax activation
#         output = self.sigmoid(output)

        return output
