import argparse
import os

import torch
import torch.nn.functional as F

import dataset as module_dataset
import model.model as module_arch
import utils.metric as module_metric

from utils.setup import setup_data_loaders
from utils.setup import SetupRun
from utils.logging import SetupLogger
from utils.util import read_json
from trainer.trainer import Trainer
from utils.visualization import TensorboardWriter


def run_training(run_setup: SetupRun,
                 logger_setup: SetupLogger,
                 vizualizer_setup,
                 device: str):
    # setup logger
    logger = logger_setup.get_logger('train')

    # setup dataset
    dataset = run_setup.init_obj(name='dataset',
                                 module=module_dataset,
                                 )
    logger.info(dataset)

    # setup data_loader instances
    data_loader, valid_data_loader = setup_data_loaders(dataset=dataset,
                                                        **run_setup['dataloader']['args'])
    
    # setup model architecture, then print to console
    model = run_setup.init_obj('arch', module_arch)
    logger.info(model)

    # setup function handles of loss and metrics
    criterion = run_setup.init_funct('loss', module_metric) # Loss is from metric.py
    metrics = [run_setup.init_funct(metric, module_metric) for metric in run_setup['metrics']]

    # setup optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = run_setup.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = run_setup.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # setup trainer
    trainer = Trainer(model=model,
                      criterion=criterion,
                      metric_ftns=metrics,
                      optimizer=optimizer,
                      run_setup=run_setup,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      vizualizer=vizualizer_setup
                      )

    trainer.train()


def main(config,
         log_config):
    # read configurations, hyperparameters for training and logging
    config = read_json(config)
    log_config = read_json(log_config)

    # set directories where trained model and log will be saved.
    checkpoint_dir = os.path.join('experiments', config['outdir'], 'checkpoint')
    log_dir = os.path.join('experiments', config['outdir'], 'log')

    run_setup = SetupRun(config=config,
                         checkpoint_dir=checkpoint_dir)

    log_setup = SetupLogger(config=log_config,
                            log_dir=log_dir)

    cfg_trainer = run_setup['trainer']['args']
    vizualizer_setup = TensorboardWriter(log_dir, cfg_trainer['tensorboard'])

    # run training process
    run_training(run_setup, log_setup, vizualizer_setup=vizualizer_setup, device=config['device'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--run-dir', default=None, type=str,
                        help='name of run directory. If it is None, the current date and time will be used')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='cpu', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-l', '--log-config', default='logger_config.json', type=str,
                        help='log config file path (default: logger_config.json)')
    args = parser.parse_args()

    main(config=args.config,
         log_config=args.log_config,
         )