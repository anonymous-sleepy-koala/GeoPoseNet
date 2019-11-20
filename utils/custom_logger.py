import numpy as np
import torch
import os, sys
import ntpath
import time, datetime
from utils.logger_utils import *
from subprocess import Popen, PIPE
import torchvision
# from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def create_folder(opt):
    # current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    base_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(base_dir): 
        os.makedirs(base_dir)

    train_output_dir = os.path.join(base_dir, 'images_train')
    val_output_dir = os.path.join(base_dir, 'images_val')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    return base_dir, train_output_dir, val_output_dir


class Logger():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses 'tensorboardX' for display
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.name = opt.name
        self.saved = True

        self.base_dir, self.train_dir, self.val_dir = create_folder(opt)

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            from tensorboardX import SummaryWriter
            import time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.base_dir, 'runs', timestr)
            if not os.path.exists(path):
                os.makedirs(path)
            self.writer = SummaryWriter(path)

        # create a logging file to store training losses
        self.log_name = os.path.join(self.base_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, len_loader):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        batches_done = epoch * len_loader + iters
        epochs_total = (self.opt.epoch_end - self.opt.epoch_start + 1)
        batches_left = epochs_total * len_loader - batches_done

        time_left = datetime.timedelta(seconds=batches_left * t_comp)

        message = '[epoch: %d/%d] [iters: %d/%d] [time: %.3f] [ETA: %s] ' % \
            (epoch, epochs_total, iters, len_loader, t_comp, time_left)

        for k, v in losses.items():
            message += '[%s %.3f] ' % (k, v)
            if self.opt.display_id > 0:
                self.writer.add_scalar('train/'+k, v, batches_done)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_metrics(self, epoch, iters, len_loader, metrics, is_train=True):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        batches_done = epoch * len_loader + iters
 
        if is_train:
            epochs = (self.opt.epoch_end - self.opt.epoch_start + 1)
        else:
            epochs = 0
        message = '[epoch: %d/%d] [iters: %d/%d] ' % \
            (epoch, epochs, iters, len_loader)

        for key in metrics.metrics:
            message += metrics.message(key) 
            message += '\n\t\t\t\t'
            if self.opt.display_id > 0:
                if is_train:
                    self.writer.add_scalar('train/'+key+'/rmse', metrics.metrics[key].rmse, batches_done)
                else:
                    self.writer.add_scalar('test/'+key+'/rmse', metrics.metrics[key].rmse, batches_done)

        print(message, end="\r")  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_statistics(self, epoch, iters, len_loader, results, is_train=True):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        batches_done = epoch * len_loader + iters
        epochs = (self.opt.epoch_end - self.opt.epoch_start + 1)
        
        message = 'test: [epoch: %d/%d] [iters: %d/%d] ' % \
            (epoch, epochs, iters, len_loader)

        for key, val in results.items():
            message += '[%s: %f]' % (key, val) 
            message += '\n\t\t\t\t'
            if self.opt.display_id > 0:
                if is_train:
                    self.writer.add_scalar('train/'+key, val, batches_done)
                else:
                    self.writer.add_scalar('test/'+key, val, batches_done)

        print(message, end="\r")  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def save_and_display_images(self, stage, epoch, iters, len_loader, rgb_images, depth_images):
        """save generated images; also display the images on tensorboard

        Parameters:
            stage (str) -- 'train' or 'test'
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            len_loader (int) -- total number of images in the dataset
            rgb_images (list) -- rgb images
            depth_images (list) -- depth_images
        """
        bs = min(rgb_images[0].size(0), 2)
        batches_done = epoch * len_loader + iters
        output_dir = self.train_dir if stage=='train' else self.val_dir

        img_sample = torch.cat((
            scale_to_255(rgb_images[0][:bs,:,:,:].data.cpu()),
            colorize_depthmap_batch(depth_images[0][:bs,:,:]),
            colorize_depthmap_batch(depth_images[1][:bs,:,:]),
        ), 0)
        torchvision.utils.save_image(img_sample,
            os.path.join(output_dir, '%02d-%05d.png' % (epoch, iters)),
            nrow=1, normalize=True)

        if self.display_id > 0:    
            img = torchvision.utils.make_grid(img_sample, nrow=bs, normalize=True)
            self.writer.add_image(stage, img, batches_done)
            # self.writer.add_image(stage + '/rgb_masked', rgb_images[1], batches_done)
            # self.writer.add_image(stage + '/depth', depth_images[0], batches_done)
            # self.writer.add_image(stage + '/depth_pred', depth_images[1], batches_done)
