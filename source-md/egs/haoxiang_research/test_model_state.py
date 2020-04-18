#!/usr/bin/env python3

import sys
#sys.path.append('/home4/md510/w2019a/espnet_20191001/espnet/espnet')
import logging
import argparse
import json
import os
import torch
#from espnet.asr.pytorch_backend.asr_init import load_trained_model

# * -------------------- general -------------------- *
def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).
    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.
    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.
    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)

import importlib
def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class
    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ':' not in import_path:
        raise ValueError(
            'import_path should be one of {} or '
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            '{}'.format(set(alias), import_path))
    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, objname)

def torch_load(path, model):
    """Load torch model states.
    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.
    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    #del model_state_dict

def load_trained_model(model_path):
    """Load the trained model for recognition.
    Args:
        model_path(str): Path to model.***.best
    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), 'model.json'))

    logging.info('reading model parameters from ' + model_path)

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)
    torch_load(model_path, model)

    return model, train_args
# pretrain model
#model_path="exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.last10.avg.best"

# run_1b
#model_path="/home4/md510/w2019a/espnet-recipe/asru2019/run_1b/exp/train_trn_pytorch_train_hkust_conv1d_statistic/results/model.val5.avg.best"
#model, train_args = load_trained_model(model_path)
#print(model,train_args)
#for param_tensor in model.state_dict():
#      print("md note: updateable parameters size of model is: " + str(param_tensor) + ' : ' + str(model.state_dict()[param_tensor].size()))

#path="exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.last10.avg.best"
#model=model.last10.avg.best
#model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
#print(model.module.load_state_dict(model_state_dict))
#for param_tensor in model.state_dict():
#      print("md note: updateable parameters size of model is: " + str(param_tensor) + ' : ' + str(model.state_dict()[param_tensor].size()))

# pretrain model
model_path="exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best"
#model_path="exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.last10.avg.best"
model, train_args = load_trained_model(model_path)
torch_load(model_path, model)
#model.load_state_dict(model_state_dict)
print(model,train_args)
for param_tensor in model.state_dict():
      print("md note: updateable parameters size of model is: " + str(param_tensor) + ' : ' + str(model.state_dict()[param_tensor].size()))
#model.load_state_dict(model_state_dict)
#for param_tensor in model.state_dict():
#      print("md note: updateable parameters size of model is: " + str(param_tensor) + ' : ' + str(model.state_dict()[param_tensor].size()))



