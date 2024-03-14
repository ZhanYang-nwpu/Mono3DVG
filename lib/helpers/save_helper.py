import os
import torch

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result, 'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def load_detr(model, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from MonoDETR '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)

        new_state_dict = {}
        for key, value in checkpoint['model_state'].items():
            # if 'transformer.decoder' in key or 'class_embed' in key or 'bbox_embed' in key\
            #     or 'dim_embed_3d' in key or 'angle_embed' in key or 'depth_embed' in key :
            #     continue
            if 'transformer' in key:
                new_key = key.replace('transformer', 'mono3dvg_transformer')
                if 'encoder.layers.' in new_key and '.self_attn' in new_key:
                    new_key = new_key.replace('self_attn', 'msdeform_attn')
            elif 'depth_predictor.classifier' in key:
                new_key = key.replace('classifier', 'depth_classifier')
            elif 'depth_predictor.encoder_proj' in key:
                new_key = key.replace('encoder_proj', 'depth_encoder')
            elif 'depth_predictor.depth_embed' in key:
                new_key = key.replace('depth_embed', 'depth_pos_embed')
            elif 'mean_size' in key or 'class_embed' in key or 'query_embed' in key:
                continue
            else:
                new_key = key

            new_state_dict[new_key] = value

        if model is not None and checkpoint['model_state'] is not None:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict,strict=False)
            # print('Missing keys when loading MonoDETR model:')
            # print(missing_keys)
        logger.info("==> Done")
    else:
        raise FileNotFoundError

def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)

        new_state_dict = {}
        for key, value in checkpoint['model_state'].items():
            if 'depthaware_transformer' in key:
                new_key = key.replace('depthaware', 'mono3dvg')
            else:
                new_key = key
            new_state_dict[new_key] = value

        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(new_state_dict,strict=False)
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch
