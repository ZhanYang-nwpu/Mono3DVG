import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.mono3drefer.mono3drefer_dataset import Mono3DRefer_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=0):
    # perpare dataset
    if cfg['type'] == 'Mono3DRefer':
        train_set = Mono3DRefer_Dataset(split=cfg['train_split'], cfg=cfg)
        val_set = Mono3DRefer_Dataset(split=cfg['val_split'], cfg=cfg)
        test_set = Mono3DRefer_Dataset(split=cfg['test_split'], cfg=cfg)
        print('train, val, test: ',len(train_set), len(val_set), len(test_set))
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False)

    val_loader = DataLoader(dataset=val_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, val_loader, test_loader
