"""
Utility functions for cascaded model training and evaluation.
"""
import libs.dataset.h36m.data_utils as data_utils
import libs.model.modelegopse as model
from libs.utils.utils import compute_similarity_transform

import torch.nn.functional as F
import torch
import numpy as np
import logging
import common.evaluate as evaluation
def train_cascade(train_dataset, eval_dataset, opt):
    """
    Train a cascade of deep neural networks that lift 2D key-points to 3D pose.
    """
    # initialize an empty cascade
    cascade = model.get_cascade()
    stage_record = []
    input_size = 34
    output_size = 51
    # train each deep learner in the cascade sequentially
    for stage_id in range(opt.num_stages):
        # initialize a single deep learner
        stage_model = model.get_model(stage_id + 1,
                                      refine_3d=opt.refine_3d,
                                      norm_twoD=opt.norm_twoD,
                                      num_blocks=opt.num_blocks,
                                      input_size=input_size,
                                      output_size=output_size,
                                      linear_size=opt.linear_size,
                                      dropout=opt.dropout,
                                      leaky=opt.leaky)
        # record the stage number
        train_dataset.set_stage(stage_id+1)
        eval_dataset.set_stage(stage_id+1)
        # move the deep learner to GPU
        if opt.cuda:
            device_ids = [0, 1]  # id为0和1的两块显卡
            # model = torch.nn.DataParallel(model, device_ids=device_ids)
            stage_model = stage_model.cuda()

        # prepare the optimizer and learning rate scheduler
        optim, sche = model.prepare_optim(stage_model, opt)

        # train the model
        record = train(train_dataset,
                       stage_model,
                       optim,
                       sche,
                       opt)
        stage_model = record['model']
        # record the training history
        stage_record.append((record['batch_idx'], record['loss']))
        # update current estimates and regression target
        train_dataset.stage_update(stage_model, opt)
        eval_dataset.stage_update(stage_model, opt)

        # update evaluation datasets for each action
        # put the trained model into the cascade
        cascade.append(stage_model.cpu())

        # release memory
        del stage_model
    return {'cascade':cascade, 'record':stage_record}



def logger_print(epoch,
                 batch_idx,
                 batch_size,
                 total_sample,
                 total_batches,
                 loss
                 ):
    """
    Log training history.
    """
    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
            epoch,
            batch_idx * batch_size,
            total_sample,
            100. * batch_idx / total_batches,
            loss.data.item())
    logging.info(msg)
    return

def train(train_dataset,
          model,
          optim,
          sche,
          opt,
          plot_loss=False):
    """
    Train a single deep learner.
    """
    eval_body = evaluation.EvalBody()
    eval_upper = evaluation.EvalUpperBody()
    eval_lower = evaluation.EvalLowerBody()
    x_data = []
    y_data = []
    if plot_loss:
        import matplotlib.pyplot as plt
        # plot loss curve during training
        ax = plt.subplot(111)
        lines = ax.plot(x_data, y_data)
        plt.xlabel('batch')
        plt.ylabel('training loss')
    for epoch in range(1, 20):
        model.train()
        # update the learning rate according to the scheduler
        sche.step()
        # data loader
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_threads
                                                   )
        for batch_idx, batch in enumerate(train_loader):
            data = batch[0]
            target = batch[1]
            data = data.to(torch.float32)
            target = target.to(torch.float32)
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            # erase all computed gradient
            optim.zero_grad()
            # forward pass to get prediction
            prediction = model(data)
            # compute loss
            loss = F.mse_loss(prediction, target)
            # smoothed l1 loss function
            #loss = F.smooth_l1_loss(prediction, target)
            # compute gradient in the computational graph
            loss.backward()
            # update parameters in the model
            optim.step()
            # logging
            if batch_idx % opt.report_every == 0:
                logger_print(epoch,
                             batch_idx,
                             opt.batch_size,
                             len(train_dataset),
                             len(train_loader),
                             loss)
                pred = prediction.reshape(prediction.shape[0], -1, 3).cpu().detach().numpy()
                gt = target.reshape(target.shape[0], -1, 3).cpu().detach().numpy()
                error = np.mean(np.sqrt(np.sum((pred * 1000 - gt * 1000) ** 2, axis=2)),axis=1)
                print(np.mean(error))
            pred = prediction.reshape(prediction.shape[0], -1, 3).cpu().detach().numpy()
            gt = target.reshape(target.shape[0], -1, 3).cpu().detach().numpy()
            eval_body.eval(pred, gt, None)
            eval_upper.eval(pred, gt, None)
            eval_lower.eval(pred, gt, None)
        res = {'FullBody': eval_body.get_results(),
                'UpperBody': eval_upper.get_results(),
                'LowerBody': eval_lower.get_results()}
        print(res)
            # optinal evaluation
        # if epoch % 50 == 0:
        #     evaluate_action_wise(action_eval_list, model, stats, opt)
    logging.info('Training finished.')
    return {'model':model, 'batch_idx':x_data, 'loss':y_data}
