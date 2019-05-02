import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset#in the file __int__.py
from models import create_model# in the file __int__.py
from utils.logger import Logger, PrintLogger
from data.util import bgr2ycbcr####add new


def main():
    # options    when you read this .py, you can read json, at the same time
    parser = argparse.ArgumentParser()#Create parser object
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')#the parameter is feed by .json file
    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
        not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():#Returns a traversable (key, value) tuple array.
        if phase == 'train':
            train_set = create_dataset(dataset_opt)#feed train set
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))#The smallest integer greater than or equal to X.train size=int(number of train_set/batch_size).
            #one epoch N iteration, one iteration batch_size batch.so totally/batch size=train size 
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            #one iters is one of the train data (batch size)
            #niter is the number of iteration
            total_epoches = int(math.ceil(total_iters / train_size))#totally epoches=int(totally iteration (number of iteration)/ train size).all of the train data
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':#this may be test? validate
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):#return enumerate object
            current_step += 1
            if current_step > total_iters:#totally iters,the number of iteration
                break

            # training#####################################################################importance part, if all of the data processining is understood, you should read the model part
            model.feed_data(train_data)# take the SR_model as example
            model.optimize_parameters(current_step)
            ##############################################################################

            time_elapsed = time.time() - start_time
            start_time = time.time()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving the model at the end of iter {:d}.'.format(current_step))
                model.save(current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                print('---------- validation -------------')
                start_time = time.time()

                ###################caculate the PSNR
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    #print(val_data['HR'].size())
                    #print(val_data['LR'].size())
                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['HR'])  # uint8
                    ########################
                    ###################################ns_img=util.tensor2img(visuals['ns']) 

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(sr_img, save_img_path)
                    #########################################3util.save_img(ns_img, save_img_path)

                    # calculate PSNR
                    # crop_size = opt['scale']
                    # #print("sr_img:{}".format(sr_img.shape))
                    # #print("gt_img:{}".format(gt_img.shape))
                    # cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    #####################################################################
                    crop_size = opt['crop_scale']
                    if crop_size>0:
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    else:
                        cropped_gt_img = gt_img.copy()
                        cropped_sr_img = sr_img.copy()
                    ####################################################################


                    avg_psnr += util.psnr(cropped_sr_img, cropped_gt_img)
                    #cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                    #cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)
                    #avg_psnr +=util.psnr(cropped_sr_img_y, cropped_gt_img_y)

                avg_psnr = avg_psnr / idx
                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                print_rlt['psnr'] = avg_psnr
                logger.print_format_results('val', print_rlt)
                print('-----------------------------------')

            # update learning rate
            model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
