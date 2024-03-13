import torch
import numpy as np
import logging
import random

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union

def create_logger(log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True