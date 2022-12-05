#!/usr/bin/env python3

import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json
from ..train import policy

# model_path = "/home/amsl/catkin_ws/src/model_base_planner/logs/20221027T175606.404775/"
# model_path = "/home/amsl/catkin_ws/src/model_base_planner/logs/20221110T090312_test.088438/"

class Badgr():
    def __init__(self, model_path):

        args_json = open(os.path.join(model_path,"args.json"), 'r')
        args_json = json.load(args_json)

        self.input_size = args_json["input_size"]
        self.batch_size = 10
        self.encoder_name = args_json["encoder_name"]
        self.linear_cost = args_json["linear_cost"]

        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(0))
        else:
            self.device = torch.device('cpu')

        pi = torch.acos(torch.zeros(1)).item() * 2

        if(self.linear_cost):
            self.model = policy.PolNetBADGRWithLinearCost(self.batch_size, self.encoder_name, self.freeze, image_shape=(3, self..input_size, self.input_size), action_shape=2).to(device)
            self.model.eval()
        else:
            self.model = policy.PolNetBADGR(self.batch_size, self.encoder_name, self.freeze, image_shape=(3, self.input_size, self.input_size), action_shape=2).to(device)
            self.model.eval()

        self.model.load_state_dict(torch.load(
            os.path.join(model_path, 'models', 'best_policy.pth'),map_location=self.device))


    def prediction(self, img, acs, overshoot_length, batch_size):
        self.overshoot_length = overshoot_length
        self.batch_size = batch_size
        with torch.no_grad():
            img = img.permute(1, 0, 2, 3, 4).to(self.device)
            acs = acs.transpose(0,1).to(self.device,non_blocking=True)

            col, pose, yaw= self.model(img, acs)

            del img

        return col, pose, yaw

if __name__ == "__main__":
    img = torch.randint(0, 255, (10, 1, 3,224, 224), dtype=torch.float32)
    acs = torch.randint(0, 1, (10, 8, 2), dtype=torch.float32)
    model_path = "/home/amsl/catkin_ws/src/badgr//"
    badgr = Badgr()
    col,pos,yaw = badgr.prediction(img,acs, 8, 10)
    print(col)

