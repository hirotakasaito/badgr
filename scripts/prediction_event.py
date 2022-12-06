#!/usr/bin/env python3

import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json
import sys
from policy import PolNetBADGRWithLinearCost, PolNetBADGRWithoutBumpy

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
        self.freeze= args_json["freeze"]

        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(0))
        else:
            self.device = torch.device('cpu')

        pi = torch.acos(torch.zeros(1)).item() * 2

        if(self.linear_cost):
            self.model = PolNetBADGRWithLinearCost(self.batch_size, self.encoder_name, self.freeze, image_shape=(3, self.input_size, self.input_size), action_shape=2).to(self.device)
            self.model.eval()
        else:
            self.model = PolNetBADGRWithoutBumpy(self.batch_size, self.encoder_name, self.freeze, image_shape=(3, self.input_size, self.input_size), action_shape=2).to(self.device)
            self.model.eval()

        self.model.load_state_dict(torch.load(
            os.path.join(model_path, 'models', 'best_policy.pth'),map_location=self.device))


    def prediction(self, img, acs):
        with torch.no_grad():
            # img = img.permute(1, 0, 2, 3, 4).to(self.device)
            img = img.to(self.device)
            # acs = acs.transpose(0,1).to(self.device,non_blocking=True)
            acs = acs.to(self.device,non_blocking=True)

            col, pose = self.model(img, acs)

            del img

        return col, pose

if __name__ == "__main__":
    img = torch.randint(0, 255, (10, 3,224, 224), dtype=torch.float32)
    acs = torch.randint(0, 1, (10,8, 2), dtype=torch.float32)
    model_path = "/home/amsl/catkin_ws/src/badgr/logs/221113000000/"
    badgr = Badgr(model_path)
    col,pos = badgr.prediction(img,acs)
    print(pos.shape)

