import torch
import gdown
import argparse
import torch.nn as nn
import subprocess
from torchvision import  models, transforms
import zipfile
import wandb
import numpy as np
from Screw_train import train
from Screw_test import predict
from Screw_test_w_class import predict_w_class





config ={
    'seed': 1234,      
    'train_ratio': 0.8,   
    'epochs': 500,    
    'batch_size': 16,
    'learning_rate': 1e-4,
    # early stop set to 10 because i already test all 500epoch for seed 1234, will not get better after epoch around 15
    'early_stop': 10,    
    'save_path': './models/model.ckpt', 
    'Screw_Img_path' : "./Screw_Img"
}


#fixes random number generator seeds for reproducibility
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# tried different transform parameters, the result gets better when we increase the contrast of image
def myTransform():
    transform = transforms.Compose([ 
        transforms.ToTensor(),           
        # transforms.Normalize(            
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )

        # transforms.Normalize(            
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5]
        # )

        transforms.Normalize(            
            mean=[0.2, 0.2, 0.2],
            std=[0.7, 0.7, 0.7]
        )
    ])
    return transform
# use restnet18 as the pretrained model
class Screw_ResNet(nn.Module):
    def __init__(self):
        super(Screw_ResNet, self).__init__()
        self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        Screw_logits = self.model(x)
        Screw_logits = Screw_logits.squeeze(1)
        return torch.sigmoid(Screw_logits)
    
def main():

    subprocess.call("conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia", shell=True)
    subprocess.call("pip install -r requirement.txt", shell=True)

    ##################### please log in to your own wandb account before hand to log the metrics, else you will need to comment all the lines related to wandb in all the .py files #####################
    wandb.init(project='Corpy Assignment Issac',config=config)
    
    url = "https://drive.google.com/file/d/11ozVs6zByFjs9viD3VIIP6qKFgjZwv9E/view?usp=sharing"
    output = "ScrewImg.zip"

    ###### we can comment out the next line if we already downloaded the training data, ex.second round training/testing #####
    gdown.download(url,output,fuzzy=True)
    zip_file_path = "./ScrewImg.zip"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(config['Screw_Img_path'])
        
    same_seed(config['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = Screw_ResNet().to(device)
    wandb.watch(model)
    wandb.log({"model": wandb.pytorch.model_to_params(model)})
    parser = argparse.ArgumentParser(description="Choose training w/or prediction function")
    parser.add_argument('--functions', nargs='*', choices=['train', 'predict', 'predict_w_class'], default=None, help='Choose functions to run, for example: "python main.py --functions train predict_w_class"')
    args = parser.parse_args()    
    

    if args.functions is None:
        # train for training
        train(model,myTransform(),config,device)

        #predict_w_class outputs the accuracy/f1_score/recall/precision for all the test dataset with my manual true label, default use the best model
        predict_w_class(model,myTransform(),device)

        #predict for visualizing each image with prediction in the dataset containing no true label, default use the best model
        predict(model,myTransform(),config,device)
    
    else:
        for function in args.functions:
            if function == 'train':
                train(model,myTransform(),config,device)
            elif function == 'predict':
                predict(model,myTransform(),config,device)
            elif function == 'predict_w_class':
                predict_w_class(model,myTransform(),device)

    
if __name__ == '__main__':
    main()
    

