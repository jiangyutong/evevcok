"""
3D pose estimation based on 2D key-point coordinates as inputs.
Author: Shichao Li
Email: nicholas.li@connect.ust.hk
"""

import logging
import os
import sys
sys.path.append("../")
import copy
import torch
import numpy as np

import libs.parser.parse as parse
import libs.utils.utils as utils
import libs.dataset.h36m.data_utils11 as data_utils

import libs.trainer.trainer as trainer

def main():
    # logging configuration
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s"
                        )

    # parse command line input
    opt = parse.parse_arg()

    # Set GPU
    opt.cuda = opt.gpuid >= 0
    if opt.cuda:
        torch.cuda.set_device(opt.gpuid)
    else:
        logging.info("GPU is disabled.")

    # dataset preparation
    train_dataset, eval_dataset, stats, action_eval_list = \
    data_utils.prepare_dataset(opt)
    train_dataset2 = copy.deepcopy(train_dataset)
    eval_dataset2 = copy.deepcopy(eval_dataset)
    stats2 = copy.deepcopy(stats)
    action_eval_list2 = copy.deepcopy(action_eval_list)
    if opt.train:
        # train a cascaded 2D-to-3D pose estimation model
        record = trainer.train_cascade(train_dataset,
                                       eval_dataset,
                                       stats,
                                       action_eval_list,
                                       opt
                                       )
        save_dir = utils.save_ckpt(opt, record, stats)

    if opt.visualize:
        # visualize the inference results of a pre-trained model
        cascade, stats = utils.load_ckpt(opt, save_dir)
        if opt.cuda:
            cascade.cuda()
        utils.visualize_cascade(eval_dataset, cascade, stats, opt)
    train_dataset, eval_dataset, stats, action_eval_list = train_dataset2, eval_dataset2, stats2, action_eval_list2
    if opt.evaluate:
        # evalaute a pre-trained cascade
        cascade, stats = utils.load_ckpt(opt, save_dir)
        trainer.evaluate_cascade(cascade,
                                 eval_dataset,
                                 stats,
                                 opt,
                                 action_wise=opt.eval_action_wise,
                                 action_eval_list=action_eval_list
                                 )


if __name__ == "__main__":
    main()