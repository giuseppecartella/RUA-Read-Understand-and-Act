# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
import vision_transformer
from .ViT import VisionTransformer
from .vit_our import ViT
from PIL import Image
from vision_transformer import timm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Cam():
    def __init__(self):
        #load model
        self.CLS2IDX = {0: 'No_signal', 1: 'go_ahead', 2: 'Left', 3: 'right', 4: 'stop' }
        model_path_cnn = 'vision_transformer/model_cnn.pt'
        model_path_vit = 'vision_transformer/model_new.pt'

        self.model_cnn = timm.create_model('resnet18', num_classes=5)
        self.model_vit = VisionTransformer(embed_dim=192, num_heads=3, num_classes=5)
     
        checkpoint_cnn = torch.load(model_path_cnn, map_location=torch.device('cpu'))
        checkpoint_vit = torch.load(model_path_vit, map_location=torch.device('cpu'))
        self.model_cnn.load_state_dict(checkpoint_cnn['model_state_dict'])
        self.model_cnn.eval()

        self.model_vit.load_state_dict(checkpoint_vit['model_state_dict'])
        self.model_vit.eval()


    def apply_transformations(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        return transform(image)

    # create heatmap from mask on image
    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    
    def generate_visualization(self, original_image, class_index=None):
        transformer_attribution, prediction = self.generate_relevance(original_image.unsqueeze(0), index=class_index)
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
        vis = self.show_cam_on_image(image_transformer_attribution, transformer_attribution)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis, prediction

    def print_top_classes(self, predictions, **kwargs):    
        # Print Top-5 predictions
        prob = torch.softmax(predictions, dim=1)
        class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
        max_str_len = 0
        class_names = []
        for cls_idx in class_indices:
            class_names.append(self.CLS2IDX[cls_idx])
            if len(self.CLS2IDX[cls_idx]) > max_str_len:
                max_str_len = len(self.CLS2IDX[cls_idx])
    
        print('Top 5 classes:')
        for cls_idx in class_indices:
            output_string = '\t{} : {}'.format(cls_idx, self.CLS2IDX[cls_idx])
            output_string += ' ' * (max_str_len - len(self.CLS2IDX[cls_idx])) + '\t\t'
            output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
            print(output_string)

    # rule 5 from paper
    def avg_heads(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    # rule 6 from paper
    def apply_self_attention_rules(self, R_ss, cam_ss):
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def generate_relevance(self, input, index=None):
        output_cnn = self.model_cnn(input)
        output = self.model_vit(input)

        print("Output cnn: ", torch.argmax(output_cnn))
        print("Output vit: ", torch.argmax(output))

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot* output)
        self.model_vit.zero_grad()
        one_hot.backward(retain_graph=True)

        num_tokens = self.model_vit.blocks[0].attn.get_attention_map().shape[-1]
        R = torch.eye(num_tokens, num_tokens)
        for blk in self.model_vit.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            cam = self.avg_heads(cam, grad)
            R += self.apply_self_attention_rules(R, cam)

        return R[0, 1:], output_cnn