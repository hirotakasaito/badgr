import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np

from encoder import Encoder
from model_utils import *

class PolNetBC(nn.Module):
    def __init__(self, batch_size, encoder_name, is_pretrained, input_shape=(3, 256, 256)):
        super(PolNetBC,self).__init__()
        self.batch_size = batch_size
        self.input_channel = input_shape[0]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        self.encoder = set_encoder(encoder_name, is_pretrained)

        self.linear1  = nn.Linear(in_features=self.check_encoder_out_put_size(), out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=512)
        self.linear4 = nn.Linear(in_features=512, out_features=2)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)

        return x

    def check_encoder_out_put_size(self):
        x = torch.FloatTensor(10, self.input_channel, self.input_width, self.input_height)
        out = self.encoder(x)
        return out.size(1)*out.size(2)*out.size(3)

class PolNetBCwithGoal(nn.Module):
    def __init__(self, batch_size, encoder_name, is_pretrained, input_shape=(3, 256, 256), goal_shape=3):
        super(PolNetBCwithGoal,self).__init__()
        self.batch_size = batch_size
        self.input_channel = input_shape[0]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        self.encoder = set_encoder(encoder_name, is_pretrained)

        self.linear1  = nn.Linear(in_features=self.check_encoder_out_put_size()+goal_shape, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=512)
        self.linear4 = nn.Linear(in_features=512, out_features=2)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()

    def forward(self, x, y):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, y], dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)

        return x

    def check_encoder_out_put_size(self):
        x = torch.FloatTensor(10, self.input_channel, self.input_width, self.input_height)
        out = self.encoder(x)
        return out.size(1)*out.size(2)*out.size(3)

class PolNetBCRNN(nn.Module):
    def __init__(self, batch_size, encoder_name, is_pretrained, input_shape=(3, 256, 256)):
        super(PolNetBCRNN,self).__init__()
        self.batch_size = batch_size
        self.input_channel = input_shape[0]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        self.encoder = set_encoder(encoder_name, is_pretrained)

        self.linear1  = nn.Linear(in_features=self.check_encoder_out_put_size(), out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=512)
        self.linear4 = nn.Linear(in_features=512, out_features=512)
        self.rnn = nn.LSTM(input_size=512, hidden_size=2, num_layers=1, batch_first=True)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()

    def forward(self, x):
        batch_size, timesteps, H, W, C = x.size()
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = x.view(batch_size, timesteps, -1)
        x , (h_n, h_c) = self.lstm(x)

        return x[:,-1,:]

    def check_encoder_out_put_size(self):
        x = torch.FloatTensor(10, self.input_channel, self.input_width, self.input_height)
        out = self.encoder(x)
        return out.size(1)*out.size(2)*out.size(3)

class PolNetBADGR(nn.Module):
    def __init__(self, batch_size, encoder_name, is_pretrained, image_shape=(3, 256, 256), action_shape=2):
        super(PolNetBADGR,self).__init__()
        self.batch_size = batch_size
        self.input_channel = image_shape[0]
        self.input_width = image_shape[1]
        self.input_height = image_shape[2]

        #image process layers
        self.encoder = set_encoder(encoder_name, is_pretrained)
        self.linear1  = nn.Linear(in_features=self.check_encoder_out_put_size(), out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=128)

        #action process layers
        self.linear5 = nn.Linear(in_features=2, out_features=16)
        self.linear6 = nn.Linear(in_features=16, out_features=16)

        #e_col process layers
        self.linear7 = nn.Linear(in_features=64, out_features=32)
        self.linear8 = nn.Linear(in_features=32, out_features=1)

        #position process layers
        self.linear9 = nn.Linear(in_features=64, out_features=32)
        self.linear10 = nn.Linear(in_features=32, out_features=3)

        #rnn
        self.rnn = nn.LSTM(input_size=16, hidden_size=64, num_layers=1, batch_first=True)

        #activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, image, action):
        action_batch_size, timesteps, _ = action.size()

        #process oveserved image
        #image = image[:,0,:,:,:]
        #image = image.permute(0, 3, 1, 2)
        #image /= 255
        x = self.encoder(image)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)

        #process action sequence
        #action = action.view(action_batch_size * timesteps, -1)
        y = self.relu(self.linear5(action))
        y = self.relu(self.linear6(y))

        #process rnn input
        #y = y.view(action_batch_size, timesteps, -1)

        h_0, c_0 = torch.chunk(x.contiguous(), 2, dim=1)

        rnn_output, (h_n, h_c) = self.rnn(y, (h_0.contiguous().unsqueeze(0), c_0.contiguous().unsqueeze(0)))

        #process rnn output
        #rnn_output = rnn_output.reshape(action_batch_size * timesteps, -1)

        e_col = self.relu(self.linear7(rnn_output))
        e_col = self.linear8(e_col)
        #e_col = e_col.view(action_batch_size*timesteps, -1)

        position = self.relu(self.linear9(rnn_output))
        position = self.linear10(position)
        yaw = self.process_yaw(position[:,:,-1])#.reshape(action_batch_size * timesteps, 1)

        #print(position.size())
        #print(yaw.size())
        #position = torch.cat([position[:,:,:-1], yaw], dim=0)
        position[:,:,-1] = yaw

        #position = position.view(action_batch_size*timesteps, -1)

        e_bumpy = self.relu(self.linear7(rnn_output))
        e_bumpy = self.linear8(e_bumpy)

        #outout shape:(batch, sequance_length, 1 or position_dim)
        return e_col, position , e_bumpy

    def process_yaw(self, yaw):
        return self.tanh(yaw) * self.pi

    def check_encoder_out_put_size(self):
        x = torch.FloatTensor(10, self.input_channel, self.input_width, self.input_height)
        out = self.encoder(x)
        return out.size(1)*out.size(2)*out.size(3)

class PolNetBADGRWithLinearCost(nn.Module):
    def __init__(self, batch_size, encoder_name, is_pretrained, image_shape=(3, 256, 256), action_shape=2):
        super(PolNetBADGRWithLinearCost,self).__init__()
        self.batch_size = batch_size
        self.input_channel = image_shape[0]
        self.input_width = image_shape[1]
        self.input_height = image_shape[2]

        #image process layers
        self.encoder = set_encoder(encoder_name, is_pretrained)
        self.linear1  = nn.Linear(in_features=self.check_encoder_out_put_size(), out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=128)

        #action process layers
        self.linear5 = nn.Linear(in_features=2, out_features=16)
        self.linear6 = nn.Linear(in_features=16, out_features=16)

        #e_col process layers
        self.linear7 = nn.Linear(in_features=64, out_features=32)
        self.linear8 = nn.Linear(in_features=32, out_features=1)

        #position process layers
        self.linear9 = nn.Linear(in_features=64, out_features=32)
        self.linear10 = nn.Linear(in_features=32, out_features=3)

        #rnn
        self.rnn = nn.LSTM(input_size=16, hidden_size=64, num_layers=1, batch_first=True)

        #activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, image, action):
        action_batch_size, timesteps, _ = action.size()

        #process oveserved image
        #image = image[:,0,:,:,:]
        #image = image.permute(0, 3, 1, 2)
        #image /= 255
        x = self.encoder(image)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)

        #process action sequence
        #action = action.view(action_batch_size * timesteps, -1)
        y = self.relu(self.linear5(action))
        y = self.relu(self.linear6(y))

        #process rnn input
        #y = y.view(action_batch_size, timesteps, -1)

        h_0, c_0 = torch.chunk(x.contiguous(), 2, dim=1)

        rnn_output, (h_n, h_c) = self.rnn(y, (h_0.contiguous().unsqueeze(0), c_0.contiguous().unsqueeze(0)))

        #process rnn output
        #rnn_output = rnn_output.reshape(action_batch_size * timesteps, -1)

        e_col = self.relu(self.linear7(rnn_output))
        e_col = self.sigmoid(self.linear8(e_col))
        #e_col = e_col.view(action_batch_size*timesteps, -1)

        position = self.relu(self.linear9(rnn_output))
        position = self.linear10(position)
        yaw = self.process_yaw(position[:,:,-1])#.reshape(action_batch_size * timesteps, 1)

        #print(position.size())
        #print(yaw.size())
        #position = torch.cat([position[:,:,:-1], yaw], dim=0)
        position[:,:,-1] = yaw

        #position = position.view(action_batch_size*timesteps, -1)

        e_bumpy = self.relu(self.linear7(rnn_output))
        e_bumpy = self.linear8(e_bumpy)

        #outout shape:(batch, sequance_length, 1 or position_dim)
        return e_col, position , e_bumpy

    def process_yaw(self, yaw):
        return self.tanh(yaw) * self.pi

    def check_encoder_out_put_size(self):
        x = torch.FloatTensor(10, self.input_channel, self.input_width, self.input_height)
        out = self.encoder(x)
        return out.size(1)*out.size(2)*out.size(3)

#class PolNetBADGRWithBumpiness(nn.Module):
#    def __init__(self, batch_size, encoder_name, is_pretrained, image_shape=(3, 256, 256), action_shape=2):
#        super(PolNetBADGRWithBumpiness,self).__init__()
#        self.batch_size = batch_size
#        self.input_channel = image_shape[0]
#        self.input_width = image_shape[1]
#        self.input_height = image_shape[2]
#
#        #image process layers
#        self.encoder = set_encoder(encoder_name, is_pretrained)
#        self.linear1  = nn.Linear(in_features=self.check_encoder_out_put_size(), out_features=256)
#        self.linear2 = nn.Linear(in_features=256, out_features=256)
#        self.linear3 = nn.Linear(in_features=256, out_features=128)
#        self.linear4 = nn.Linear(in_features=128, out_features=128)
#
#        #action process layers
#        self.linear5 = nn.Linear(in_features=2, out_features=16)
#        self.linear6 = nn.Linear(in_features=16, out_features=16)
#
#        #e_col process layers
#        self.linear7 = nn.Linear(in_features=64, out_features=32)
#        self.linear8 = nn.Linear(in_features=32, out_features=1)
#
#        #position process layers
#        self.linear9 = nn.Linear(in_features=64, out_features=32)
#        self.linear10 = nn.Linear(in_features=32, out_features=3)
#
#        #e_bumpy process layers
#        self.linear11 = nn.Linear(in_features=64, out_features=32)
#        self.linear12 = nn.Linear(in_features=32, out_features=1)
#
#        #rnn
#        self.rnn = nn.LSTM(input_size=16, hidden_size=64, num_layers=1, batch_first=True)
#
#        #activation
#        self.relu = nn.ReLU()
#        self.tanh = nn.Tanh()
#
#        self.pi = torch.acos(torch.zeros(1)).item() * 2
#
#    def forward(self, image, action):
#        action_batch_size, timesteps, _ = action.size()
#
#        #process oveserved image
#        image = image[:,0,:,:,:]
#        image = image.permute(0, 3, 1, 2)
#        image /= 255
#        x = self.encoder(image)
#        x = torch.flatten(x, 1)
#        x = self.relu(self.linear1(x))
#        x = self.relu(self.linear2(x))
#        x = self.relu(self.linear3(x))
#        x = self.linear4(x)
#
#        #process action sequence
#        action = action.view(action_batch_size*timesteps, -1)
#        y = self.relu(self.linear5(action))
#        y = self.linear6(y)
#
#        #process rnn input
#        y = y.view(action_batch_size, timesteps, -1)
#
#        h_0, c_0 = torch.chunk(x.contiguous(),2,dim=1)
#
#        rnn_output, (h_n, h_c) = self.rnn(y, (h_0.contiguous().unsqueeze(0), c_0.contiguous().unsqueeze(0)))
#
#        #process rnn output
#        rnn_output = rnn_output.reshape(action_batch_size * timesteps, -1)
#
#        e_col = self.relu(self.linear7(rnn_output))
#        e_col = self.linear8(e_col)
#
#        position = self.relu(self.linear9(rnn_output))
#        position = self.linear10(position)
#        yaw = self.process_yaw(position[:,-1]).reshape(action_batch_size * timesteps, 1)
#
#        position = torch.cat([position[:,:-1], yaw], dim=1)
#
#        position = position.view(action_batch_size*timesteps, -1)
#
#        e_bumpy = self.relu(self.linear11(rnn_output))
#        e_bumpy = self.linear12(e_bumpy)
#
#        #outout shape:(batch*sequance_length, 1 or position_dim)
#        return e_col, position, e_bumpy
#
#    def process_yaw(self, yaw):
#        return self.tanh(yaw) * self.pi
#
#    def check_encoder_out_put_size(self):
#        x = torch.FloatTensor(10, self.input_channel, self.input_width, self.input_height)
#        out = self.encoder(x)
#        return out.size(1)*out.size(2)*out.size(3)
