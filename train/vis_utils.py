import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import animation

def angle_normalize(z):
    return np.arctan2(np.sin(z), np.cos(z))

def lidar2image(lidar, width=100, height=100, max_lengh=20.0):
    img = Image.new('RGB', (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)
    scale = width*0.5/max_lengh
    x = width * 0.5
    y = height
    for idx, r in enumerate(lidar):
        r *= scale
        theta = idx/len(lidar) * np.pi
        x_ = x + r*np.cos(theta)
        y_ = y - r*np.sin(theta)
        draw.line((x,y,x_,y_),fill=(255,0,0),width=1)
    return img

def pose2image(poses, pred_poses, width=1024, height=1024, max_range=1.0):
    img = Image.new('RGB', (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)
    scale = width*0.5/max_range
    x = width * 0.5
    y = height * 0.5
    radius = 1
    pre_x = x + poses[0][0]*scale
    pre_y = x + poses[0][1]*scale
    pre_x_ = x + pred_poses[0][0]*scale
    pre_y_ = y + pred_poses[0][1]*scale
    for p, pred_p in zip(poses[1:], pred_poses[1:]):
        x_ = x + p[0]*scale
        y_ = y + p[1]*scale
        draw.line((pre_x,pre_y,x_,y_),fill=(0,0,255),width=2)
        pre_x = x_
        pre_y = y_
        x__ = x + pred_p[0]*scale
        y__ = y + pred_p[1]*scale
        draw.line((pre_x_,pre_y_,x__,y__),fill=(0,255,0),width=2)
        pre_x_ = x__
        pre_y_ = y__
    return img

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h_multi(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h(_im, im)
    return _im

def save_video_as_gif(frames, interval=150, file_name="video_prediction.gif", title=""):
    """
    make video with given frames and save as "video_prediction.gif"
    """
    plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        plt.title(title + ' \n Step %d' % (i))

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    anim.save(file_name, writer='imagemagick')