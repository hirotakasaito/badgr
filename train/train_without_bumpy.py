import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split

import tensorboardX as tbx

import argparse
import os
import datetime
import json
from tqdm import tqdm
from collections import OrderedDict
from pprint import pprint

from policy import PolNetBADGR, PolNetBADGRWithLinearCost, PolNetBADGRWithoutBumpy
from dataset import FullDataset

from vis_utils import pose2image
import numpy as np
import time

def train(model, device, dataloader, optimizer, col_loss_fn, pose_loss_fn, epoch, tb_writer):

    model.train()
    train_acc = 0
    loss_time = 1
    mean_loss=0
    mean_pose_loss=0
    mean_col_loss=0
    step = 0

    total_len = len(dataloader)
    with tqdm(dataloader, ncols=100) as pbar_train:
        for batch_i, (image, action, pose, is_collision) in enumerate(pbar_train):
            t1 = time.time()
            #if(batch_i>0):
                #print("#########################################")
                #print(t1-t2)
            image = image.permute(0,1,4,2,3)
            image = image[:,0,:,:,:]
            image = image.to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.float32)
            pose = pose.to(device, dtype=torch.float32)
            is_collision = is_collision.to(device, dtype=torch.float32)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                est_collision, est_pose  = model(image, action)

                #pose = pose.view(image.size(0)*image.size(1), -1)
                #is_collision = is_collision.view(image.size(0)*image.size(1), -1)
                #is_bumpy = is_bumpy.view(image.size(0)*image.size(1), -1)

                is_collision.unsqueeze_(2)

                pose_loss = pose_loss_fn(est_pose, pose)
                col_loss = col_loss_fn(est_collision, is_collision)
                loss = pose_loss + col_loss

                loss.backward()
                optimizer.step()
                step += 1

                mean_loss += loss.item()
                mean_pose_loss += pose_loss.item()
                mean_col_loss += col_loss.item()
                #tb_writer.add_scalar('train/loss', loss.item(), total_len*epoch+step)
                #tb_writer.add_scalar('train/pose_loss', pose_loss.item(), total_len*epoch+step)
                #tb_writer.add_scalar('train/col_loss', col_loss.item(), total_len*epoch+step)
                #tb_writer.add_scalar('train/bumpy_loss', bumpy_loss.item(), total_len*epoch+step)
            t2 = time.time()
            #print(t2 - t1)
            pbar_train.set_postfix(OrderedDict(epoch="{:>3}".format(epoch),loss="{:.4f}".format(loss.item())))

            del loss
            del pose_loss
            del col_loss
            del image
            del action
            del is_collision
            del est_collision
            torch.cuda.empty_cache()

        pose = pose[0].detach().to('cpu').numpy()
        pred_pose = est_pose[0].detach().to('cpu').numpy()
        pos_img = pose2image(pose, pred_pose)
        pos_img_np = np.asarray(pos_img).transpose(2,0,1)
    return mean_loss/step, mean_pose_loss/step, mean_col_loss/step, pos_img_np

def test_model(model, device, col_loss_fn, pose_loss_fn, bumpy_loss_fn, testloader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_pose_loss = 0
        val_col_loss = 0
        count = 0
        for batch_i, (image, action, pose, is_collision) in enumerate(testloader):

            image = image.permute(0,1,4,2,3)
            image = image[:,0,:,:,:]
            image = image.to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.float32)
            pose = pose.to(device, dtype=torch.float32)
            is_collision = is_collision.to(device, dtype=torch.float32)

            est_collision, est_pose = model(image, action)

            #pose = pose.view(image.size(0)*image.size(1), -1)
            #is_collision = is_collision.view(image.size(0)*image.size(1), -1)
            #is_bumpy = is_bumpy.view(image.size(0)*image.size(1), -1)

            is_collision.unsqueeze_(2)

            pose_loss = pose_loss_fn(est_pose, pose)
            col_loss = col_loss_fn(est_collision, is_collision)
            loss = pose_loss + col_loss
            val_loss += loss.item()
            val_pose_loss += pose_loss.item()
            val_col_loss += col_loss.item()
            count += 1

            del loss
            del pose_loss
            del col_loss
            del image
            del action
            del is_collision
            del est_collision
            torch.cuda.empty_cache()

        #print('eval_loss: %.3f' % (val_loss / count))
        pose = pose[0].detach().to('cpu').numpy()
        pred_pose = est_pose[0].detach().to('cpu').numpy()
        pos_img = pose2image(pose, pred_pose)
        pos_img_np = np.asarray(pos_img).transpose(2,0,1)

    return val_loss/count, val_pose_loss/count , val_col_loss/count, pos_img_np

def main():

    parser = argparse.ArgumentParser(description='Python script for Training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--test_ratio', default=0.9, type=float, help='split ratio')

    parser.add_argument('--dataset_dir', type=str, default='d_kan1')
    parser.add_argument('--input_size', default=224, type=int, help='size of input image')
    parser.add_argument('--test_interval', default=5, type=int, help='test interval')
    parser.add_argument('--image_log_interval', default=1, type=int, help='pose image view interval')

    parser.add_argument('--log_dir', default='/root/logs/badgr', help='dir_name of log for')
    parser.add_argument('--multi', default=False, action='store_true', help='use multi GPUs')
    parser.add_argument('--gpu_id', default='0', type=str, help='id of GPUs')
    parser.add_argument('--encoder_name', default=None, type=str, help='encoder name')
    parser.add_argument('--freeze', action='store_true', help='freeze conv layers')
    parser.add_argument('--linear_cost', action='store_true', help='use linear collision cost')
    parser.add_argument('--trans', action='store_true', help='use image data augumentation')
    parser.add_argument("--num_workers", type=int, default=4, help='number of thread for dataloader')


    args = parser.parse_args()
    date = datetime.date.today().strftime('%y%m%d%H%M%S')
    log_dir = os.path.join(args.log_dir, args.dataset_dir, date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir,'models')):
        os.mkdir(os.path.join(log_dir, 'models'))
    tb_writer = tbx.SummaryWriter(log_dir=log_dir)

    with open(os.path.join(log_dir,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))

    with open(os.path.join("./config/badgr_dirs.json")) as f:
        dataset_dirs = json.load(f)
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    dataset = FullDataset(dataset_dirs=dataset_dirs[args.dataset_dir], config_path="./config/BADGR.json", is_trans=args.trans, linear_cost=args.linear_cost)
    train_num = int(args.test_ratio*len(dataset))
    test_num = len(dataset) - train_num
    print(f"train_num: {train_num}")
    print(f"test_num : {test_num}")
    train_data, test_data = random_split(dataset, [train_num, test_num])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, pin_memory=True)

    if(args.linear_cost):
        col_loss_fn = torch.nn.MSELoss()
        model = PolNetBADGRWithLinearCost(args.batch_size, args.encoder_name, args.freeze, image_shape=(3, args.input_size, args.input_size), action_shape=2).to(device)
    else:
        col_loss_fn = torch.nn.BCEWithLogitsLoss()
        model = PolNetBADGRWithoutBumpy(args.batch_size, args.encoder_name, args.freeze, image_shape=(3, args.input_size, args.input_size), action_shape=2).to(device)

    pose_loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1000000
    test_loss = 0

    if(args.multi):
        model = torch.nn.DataParallel(model) # make parallel
        model = torch.nn.DistributedDataParallel(model)

    for epoch in range(args.epoch):

        loss, pose_loss, col_loss, pos_img_np = train(model, device, train_loader, optimizer, col_loss_fn, pose_loss_fn, epoch, tb_writer)

        tb_writer.add_scalar('train/loss', loss, epoch+1)
        tb_writer.add_scalar('train/pose_loss', pose_loss, epoch+1)
        tb_writer.add_scalar('train/col_loss', col_loss, epoch+1)
        if(epoch%args.image_log_interval==0):
            tb_writer.add_image('train/pos_img', pos_img_np, epoch+1)
        #print("############epoch finish############")

        if((epoch+1)%args.test_interval==0):
            #print("evaluate model...")
            test_loss, test_pose_loss, test_col_loss, pos_img_np = test_model(model, device, col_loss_fn, pose_loss_fn, test_loader)
            tb_writer.add_scalar('test/loss', test_loss, epoch+1)
            tb_writer.add_scalar('test/pose_loss', test_pose_loss, epoch+1)
            tb_writer.add_scalar('test/col_loss', test_col_loss, epoch+1)
            tb_writer.add_image('test/pos_img', pos_img_np, epoch+1)

            if(test_loss < best_loss):
                best_loss = test_loss
                if(args.multi):
                    torch.save(model.module.state_dict(), os.path.join(log_dir, 'models','best_policy.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(log_dir, 'models','best_policy.pth'))
            else:
                if(args.multi):
                    torch.save(model.module.state_dict(), os.path.join(log_dir, 'models','latest_policy.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(log_dir, 'models','latest_policy.pth'))

    if((args.epoch)%args.test_interval!=0):
            #print("evaluate model...")
            test_loss, test_pose_loss, test_col_loss, pos_img_np = test_model(model, device, col_loss_fn, pose_loss_fn, test_loader)
            tb_writer.add_scalar('test/loss', test_loss, epoch+1)
            tb_writer.add_scalar('test/pose_loss', test_pose_loss, epoch+1)
            tb_writer.add_scalar('test/col_loss', test_col_loss, epoch+1)
            tb_writer.add_image('test/pos_img', pos_img_np, epoch+1)

            if(test_loss < best_loss):
                best_loss = test_loss
                if(args.multi):
                    torch.save(model.module.state_dict(), os.path.join(log_dir, 'models','best_policy.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(log_dir, 'models','best_policy.pth'))
            else:
                if(args.multi):
                    torch.save(model.module.state_dict(), os.path.join(log_dir, 'models','latest_policy.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(log_dir, 'models','latest_policy.pth'))
    tb_writer.close()

if __name__ == '__main__':
    main()
