from distutils.command.config import config
import torch
import json
import os
from vision_transformer.ViT import ViT

#under "vision_transformer directory" we should put model.pt + config.json
#in order to insert best fine tuned hyperparameters


class SignsReader():
    def __init__(self):
        #load model
        root = 'vision_transformer'
        config_path = os.path.join(root, 'config.json')
        model_path = os.path.join(root, 'model.pt')
        with open(config_path) as f:
            config = json.load(f)

        #to check if our version is the best model. otherwise use finetuned version using timm
        self.model = ViT()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
