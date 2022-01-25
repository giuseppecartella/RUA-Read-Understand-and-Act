from distutils.command.config import config
import torch
import timm
import json
import os
from vision_transformer.ViT import ViT

#under "vision_transformer directory" we should put model.pt + config.json
#in order to insert best fine tuned hyperparameters
#NB: MAYBE THE CNN IS THE BEST CHOICE TO PREFER.


class SignsReader():
    def __init__(self):
        #load model
        root = 'vision_transformer'
        config_path = os.path.join(root, 'config.json')
        model_path = os.path.join(root, 'model.pt')
        
        with open(config_path) as f:
            config = json.load(f)


        self.model = timm.create_model('vit_tiny_patch16_224')
        self.model.eval()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        with open('vision_transformer/classes.json') as f:
            self.classes = json.load(f)


    def predict(self, observation):
        #forse c'e' da fare il resize dell img
        self.model.eval()
        
        prediction = self.model(observation)
        print(prediction)
        prediction = prediction.argmax()
        print(prediction)
        print('E stata predetta la classe: ' + self.classes[str(prediction)])
        return prediction



