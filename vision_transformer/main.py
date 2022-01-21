import argparse
import os
import json
import torch
from train import Trainer


if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--job_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-m', '--model_type', type=str, default='vit', choices=("cnn", "vit", "vit_trained")) #nome del modello usato
    parser.add_argument('-d', '--dataset', type=str, default='dataset') # da cambiare se vogliamo dataset no turn
    parser.add_argument('-e', '--epochs', type=int, default=300)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-drop', '--dropout', type=float, default=0.3)
    parser.add_argument('-nl', '--n_layers', type=int, default=6)
    parser.add_argument('-ft', '--first_train', action='store_true') #se voglio a True specificare solo -ft da command line(senza param)
    parser.add_argument('-m_name', '--model_name', type=str, default='model.pt') #come salviamo modello e.g 'model_0_3'
    
    args = parser.parse_args()
    config = vars(args)

    dir_name = "logs_" + config['job_name']

    config_file = os.path.join(dir_name, 'config.json')
    with open(config_file,'w') as f:
        json.dump(config, f)

    # caricamento del modello
    if config['model_type'] == 'cnn':
        import timm
        model = timm.create_model('resnet18', pretrained=True)
        model.eval()
    elif config['model_type'] == 'vit':
        from ViT import ViT
        from torchvision import datasets, transforms
        model = ViT(   
        image_size=224,
        patch_size=32,
        num_classes=5, 
        dim=128,
        depth = config['n_layers'], #args.N_layers
        heads=8,
        mlp_dim =256, 
        channels=3,
        dropout= config['dropout'],
        emb_dropout= config['dropout']
        )
    elif config['model_type'] == 'vit_trained':
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        model.eval()
    else:
        raise NotImplementedError
    
    trainer = Trainer()
    # creazione dataloader e caricamento dataset
    train_loader , test_loader = trainer.load_dataset(config)

    # training del modello
    trainer.train_model(model, train_loader ,test_loader, config)