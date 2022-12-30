import paddle
import paddle.distributed as dist
import argparse
import os
import anyconfig


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.paddle')
    parser.add_argument('--config_file', default=
        'config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=
        int, help='Use distributed training')
    args = parser.parse_args()
    return args


def main(config):
    from models import build_model, build_loss
    from data_loader import get_dataloader
    from trainer import Trainer
    from post_processing import get_post_processing
    from utils import get_metric
    if paddle.device.cuda.device_count() > 1:
        dist.init_parallel_env()
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank
    train_loader = get_dataloader(config['dataset']['train'], config[
        'distributed'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None
    criterion = build_loss(config['loss'])
    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train'
        ]['dataset']['args']['img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])
    post_p = get_post_processing(config['post_processing'])
    metric = get_metric(config['metric'])
    trainer = Trainer(config=config, model=model, criterion=criterion,
        train_loader=train_loader, post_process=post_p, metric_cls=metric,
        validate_loader=validate_loader)
    trainer.train()


if __name__ == '__main__':
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    from utils import parse_config
    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)
