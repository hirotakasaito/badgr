#!/usr/bin/env python3

import torch
import numpy as np
from prediction_event import *
import time
import rospy
import cv2
import tf
import torchvision.transforms.functional as TF
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import CompressedImage, Image

import random

class MPPIPlanner():
    """ Model Predictive Path Integral for linear and nonlinear method
    Attributes:
        history_u (list[numpy.ndarray]): time history of optimal input
    Ref:
        Nagabandi, A., Konoglie, K., Levine, S., & Kumar, V. (2019).
        Deep Dynamics Models for Learning Dexterous Manipulation.
        arXiv preprint arXiv:1909.11652.
    """
    def __init__(self,):
        super(MPPIPlanner, self).__init__()

        rospy.init_node('mppi_dwa', anonymous=True)
        self.HZ = rospy.get_param("~HZ", 10)
        self.img_height = rospy.get_param("~IMG_HEIGHT", 224)
        self.img_width = rospy.get_param("~IMG_WIDTH", 224)
        self.pred_len = rospy.get_param("~PREDITCT_TIME", 21)
        self.kappa = rospy.get_param("~KAPPA", 50)
        self.dt= rospy.get_param("~DT", 0.1)
        self.model_path = rospy.get_param("~MODEL_PATH", "/home/amsl/catkin_ws/src/badgr/logs/221113000000/")


        self.cuda = 0
        if self.cuda < 0 or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:'+str(self.cuda))
            print("GPU使用")

        # model
        # self.model = model

        self.input_size = 2
        self.action = np.zeros(2)
        self.goal = torch.rand(2, dtype=torch.float32).to(self.device)
        self.pose = np.zeros(3)
        self.local_goal = torch.rand(2, dtype=torch.float32).to(self.device)


        # mppi parameters
        self.pop_size = 0
        self.opt_dim = self.input_size * self.pred_len

        self.prev_sol = torch.zeros([self.pred_len, self.input_size], dtype=torch.float32)

        self.received_camera_data = False
        self.received_local_goal = False
        # save
        self.history_u = [torch.zeros(self.input_size)]
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        #ros
        self.trajs_pub = rospy.Publisher('/local_path/trajectories',MarkerArray, queue_size = 1)
        self.traj_pub = rospy.Publisher('/local_path/selected_trajectory', Marker, queue_size = 1)
        self.cmd_vel_pub = rospy.Publisher('/local_path/cmd_vel', Twist, queue_size = 11)

        self.camera_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.camera_callback)
        self.local_sub = rospy.Subscriber('/local_goal', PoseStamped , self.local_goal_callback)

    def local_goal_callback(self,data):
        self.local_goal[0] = data.pose.position.x
        self.local_goal[1] = data.pose.position.y
        self.received_local_goal = True

    def camera_callback(self, data):
        camera_data_img = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)
        h,w,c = camera_data_img.shape
        camera_data_img = camera_data_img[0:h,int((w-h)*0.5):w-int((w-h)*0.5),:]
        camera_data_img = cv2.resize(camera_data_img, (self.img_height, self.img_width))
        camera_data = TF.to_tensor(camera_data_img).to(device = self.device)
        self.camera_data = torch.unsqueeze(camera_data, 0)
        self.received_camera_data = True

    def trajectories_visualization(self, trajectories):
        trajectories = trajectories[:,:,0:2]
        vis_trajectories = MarkerArray()
        #print(trajecties[0])
        for i in range(self.pop_size):
            marker_data = Marker()
            marker_data.header.frame_id = "base_link"
            marker_data.header.stamp = rospy.Time.now()
            marker_data.type = 4 #LINE_STRIP

            marker_data.ns = "/local_path/trajectories"
            marker_data.id = i

            marker_data.action = Marker.ADD

            marker_data.color.r = 0.0
            marker_data.color.g = 1.0
            marker_data.color.b = 0.0
            marker_data.color.a = 0.8

            marker_data.pose.orientation.w = 1.0
            marker_data.scale.x = 0.02

            for p in trajectories[i]:
                point = Point()
                point.x = p[0].item()
                point.y = p[1].item()
                marker_data.points.append(point)


            marker_data.lifetime = rospy.Duration()

            vis_trajectories.markers.append(marker_data)
        self.trajs_pub.publish(vis_trajectories)

    def trajectory_visualization(self, trajectory):
        #print(trajectory)
        marker_data = Marker()
        marker_data.header.frame_id = "base_link"
        marker_data.header.stamp = rospy.Time.now()
        marker_data.type = 4 #LINE_STRIP
        marker_data.ns = "/local_path/selected_trajectory"
        marker_data.id = 0
        marker_data.action = Marker.ADD
        marker_data.color.r = 1.0
        marker_data.color.g = 0.0
        marker_data.color.b = 0.0
        marker_data.color.a = 0.8
        marker_data.pose.orientation.w = 1.0
        marker_data.scale.x = 0.02
        point = Point()
        pre_point = Point()
        pre_point.x = 0
        pre_point.y = 0
        yaw = 0

        v = trajectory[0,0].item()
        omega = trajectory[0,1].item()
        for a in trajectory:
            point = Point()
            _v = a[0].item()
            _omega = a[1].item()
            point.x = pre_point.x + v * self.dt * math.cos(yaw)
            point.y = pre_point.y + v * self.dt * math.sin(yaw)
            marker_data.points.append(point)
            yaw = yaw + omega * self.dt
            if(yaw < -math.pi and yaw > math.pi):
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            pre_point = point

        marker_data.lifetime = rospy.Duration()

        self.traj_pub.publish(marker_data)

    def clear_sol(self):
        """ clear prev sol
        """
        self.prev_sol = np.zeros([self.pred_len, self.input_size], dtype=np.float32)

    def calc_cost(self, curr_x, trajs, g_xs,):

        self.pop_size = trajs.shape[0]
        cmd_vel = trajs[:,:,3:]

        # est_pose = trajs[:,:,0:2]
        # est_pose = torch.from_numpy(est_pose.astype(np.float32)).clone().to(self.device)

        cmd_vel = torch.from_numpy(cmd_vel.astype(np.float32)).clone().to(self.device)

        curr_x = curr_x.repeat(self.pop_size,1,1,1)

        est_collision, est_pose = self.model.prediction(curr_x, cmd_vel)
        est_collision, _ = self.model.prediction(curr_x, cmd_vel)
        est_pose = est_pose[:,:,:2]

        # est_pose = est_pose.permute(1,0,2)
        trajs = est_pose.to('cpu').detach().numpy().copy()

        est_collision = est_collision.reshape(self.pop_size, -1)
        est_collision = torch.clamp(est_collision, min=0.02, max=1-0.02)
        cost_collision = (est_collision - 0.02) / (1. - 2 * 0.02)

        # est_pose = est_pose.reshape(self.pop_size, self.pred_len, -1)

        g_xs = g_xs.repeat(self.pop_size,self.pred_len, 1)

        dot_product = torch.sum(est_pose*g_xs, dim=2)
        a_norm = torch.norm(est_pose,dim=2)
        b_norm = torch.norm(g_xs,dim=2)

        cos_theta = dot_product / torch.max(a_norm * b_norm, torch.zeros_like(a_norm * b_norm)+1e-4)
        theta = torch.acos(torch.clamp(cos_theta, -1+1e-4, 1-1e-4))
        cost_position = (1. / self.pi) * torch.abs(theta)

        cost_position = (1. - cost_collision) * cost_position + cost_collision * 1.

        return cost_position + cost_collision, trajs

    def get_costs(self, curr_x, g_xs, trajs):
        #noised_inputs = self.get_noised_action()

        # calc cost
        # cost : Tensot of cost at each step (shape[num_candidate, seq_len])

        costs, trajs = self.calc_cost(curr_x, trajs, g_xs)

        return costs, trajs

    def get_action(self, cmd_vel, costs):
        sum_cost = torch.sum(costs, dim=1)

        rewards = -sum_cost

        # mppi update
        # normalize and get sum of reward
        # exp_rewards.shape = (N, )
        exp_rewards =torch.exp(self.kappa * (rewards - torch.max(rewards)))
        denom = torch.sum(exp_rewards) + 1e-10  # avoid numeric error

        # weight actions

        weighted_inputs = exp_rewards.unsqueeze(1).unsqueeze(2) \
                          * cmd_vel
        sol = torch.sum(weighted_inputs, 0) / denom

        # update
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]  # last use the terminal input

        # log
        self.history_u.append(sol[0])
        action = sol[0].data
        action = action.to('cpu').detach().numpy().copy()
        sol = sol.to('cpu').detach().numpy().copy()

        return action, sol

    def create_pose_cmd_vel(self,):
        x = [0.0,0.0,0.0,self.action[0],self.action[1]]
        dw = self.calc_dynamic_window(x)
        v_reso = 0.2
        w_reso = 0.2
        trajs = np.zeros((1,self.pred_len,5))
        for v in np.arange(dw[0],dw[1],v_reso):
            for y in np.arange(dw[2], dw[3], w_reso):
                traj = self.calc_trajectory(x, v, y)
                trajs = np.insert(trajs,-1,traj,axis=0)

        trajs = np.delete(trajs,-1,axis=0)
        cmd_vel = trajs[:,:,3:]

        return cmd_vel, trajs

    def process(self, model):
        r = rospy.Rate(self.HZ)
        selected_cmd_vel = Twist()
        self.local_goal[0] = 4.0
        self.local_goal[1] = 0.0
        self.model = model

        while not rospy.is_shutdown():
            if(self.received_local_goal and self.received_camera_data):
                    # self.local_goal[0] = 4.0
                    # self.local_goal[1] = 0.0

                    cmd_vel,trajs = self.create_pose_cmd_vel()
                    cmd_vel = torch.from_numpy(cmd_vel.astype(np.float32)).clone().to(self.device)
                    costs, trajs = self.get_costs(self.camera_data, self.local_goal, trajs)
                    self.action, sol = self.get_action(cmd_vel, costs)
                    planner.trajectories_visualization(trajs)
                    planner.trajectory_visualization(sol)
                    selected_cmd_vel.linear.x = self.action[0]
                    selected_cmd_vel.angular.z = self.action[1]
                    self.cmd_vel_pub.publish(selected_cmd_vel)
                    print(selected_cmd_vel)
                    print("Predict Time:{0},Trajectory:{1}".format(trajs.shape[1]*self.dt,trajs.shape[0]))
                    r.sleep()

    def __str__(self):
        return "MPPI"

    def calc_dynamic_window(self,x):
        max_v = 0.8
        min_v = 0
        max_w = 1.0
        max_d_w = 5.0
        max_acceleratoin = 5#1.5

        Vs = [min_v,max_v,-max_w,max_w]
        Vd = [x[3] - max_acceleratoin * self.dt,
              x[3] + max_acceleratoin * self.dt,
              x[4] - max_d_w * self.dt,
              x[4] + max_d_w * self.dt]

        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_trajectory(self, x, v, y):

        x = np.array(x)

        traj = np.array(x)
        time = 0
        while time < self.pred_len*self.dt:
            x = self.motion(x, [v, y], self.dt)
            traj = np.vstack((traj, x))
            time += self.dt

        traj = np.delete(traj,0,axis=0)
        return traj


    def motion(self, x, u, dt):

        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]

        return x

if __name__ == '__main__':

    planner = MPPIPlanner()
    model = Badgr(planner.model_path)
    # planner = MPPIPlanner(model)
    try:
        planner.process(model)
    except rospy.ROSInterruptException:
        pass
