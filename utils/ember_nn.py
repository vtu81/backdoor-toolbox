import torch
from torch import nn

class EmberNN(nn.Module):

    def __init__(self, n_features = 2351, num_classes=2):

        super(EmberNN, self).__init__()

        # NUM_EMBER_FEATURES = 2351
        self.n_features = n_features

        self.dense1 = nn.Linear(n_features, 4000)
        self.norm1 = nn.BatchNorm1d(4000)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(4000, 2000)
        self.norm2 = nn.BatchNorm1d(2000)
        self.drop2 = nn.Dropout(0.5)

        self.dense3 = nn.Linear(2000, 100)
        self.norm3 = nn.BatchNorm1d(100)
        self.drop3 = nn.Dropout(0.5)

        self.dense4 = nn.Linear(100, 1)

        #self.normal = StandardScaler()
        #self.model = self.build_model()
        #self.exp = None
        #lr = 0.1
        #momentum = 0.9
        #decay = 0.000001
        #opt = SGD(lr=lr, momentum=momentum, decay=decay)
        #self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def forward(self, x):

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.dense2(x)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.dense3(x)
        x = torch.relu(x)
        x = self.norm3(x)
        x = self.drop3(x)

        x = self.dense4(x)
        x = torch.sigmoid(x).reshape(-1)

        return x



class EmberNN_narrow(nn.Module):

    def __init__(self, n_features = 2351, num_classes=2):

        super(EmberNN_narrow, self).__init__()

        # NUM_EMBER_FEATURES = 2351
        self.n_features = n_features

        self.dense1 = nn.Linear(n_features, 200)
        self.norm1 = nn.BatchNorm1d(200)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(200, 100)
        self.norm2 = nn.BatchNorm1d(100)
        self.drop2 = nn.Dropout(0.5)

        self.dense3 = nn.Linear(100, 20)
        self.norm3 = nn.BatchNorm1d(20)
        self.drop3 = nn.Dropout(0.5)

        self.dense4 = nn.Linear(20, 1)

        #self.normal = StandardScaler()
        #self.model = self.build_model()
        #self.exp = None
        #lr = 0.1
        #momentum = 0.9
        #decay = 0.000001
        #opt = SGD(lr=lr, momentum=momentum, decay=decay)
        #self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def forward(self, x):

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.dense2(x)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.dense3(x)
        x = torch.relu(x)
        x = self.norm3(x)
        x = self.drop3(x)

        x = self.dense4(x)
        x = torch.sigmoid(x).reshape(-1)

        return x