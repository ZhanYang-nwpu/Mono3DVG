import warnings
warnings.filterwarnings("ignore")
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Mono3DVG Transformer for Monocular 3D Visual Grounding')
parser.add_argument('--config', default='configs/mono3dvg.yaml', help='settings of detection in yaml format')

args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['trainer']['gpu_ids'][0]

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    train_loader,val_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, loss = build_model(cfg['model'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    logger.info('###################  Mono3DVG-TR Testing  ##################')
    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger,
                    loss=loss,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    tester.test()


if __name__ == '__main__':
    main()

