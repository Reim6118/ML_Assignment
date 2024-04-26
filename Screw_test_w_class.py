import torch
from torch.utils.data import  DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import gdown
import zipfile
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_auc_score
import wandb
import pandas as pd

    

# predict the whole test set with me manually label the pngs. Output will be the f1score and overalll accuracy
def predict_w_class(model,myTransform,device):

    #Downloading dataset
    url = "https://drive.google.com/file/d/1l2MKi0PQtxIME6gmM-UVpcjMnoIj1jNQ/view?usp=sharing"
    output = "Screw_test_w_class.zip"
    gdown.download(url,output,fuzzy=True)   
    zip_file_path = "./Screw_test_w_class.zip"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("./Screw_test_w_class")
    # load test set
    Screw_test = datasets.ImageFolder("./Screw_test_w_class/test_w_label",myTransform)
    test_loader = DataLoader(Screw_test)
    #use the previous trained model
    state_dict = torch.load('./models/model_best.ckpt')
    model.load_state_dict(state_dict)
    print("Testmodel : ",model)

    model.eval()
    image_names =[]
    true_labels = []
    predicted_labels = []
    for x,y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)            

        y_numpy = y.cpu().numpy()
        pred_numpy = (pred.cpu().numpy() >= 0.5).astype(int) 
        # print out all the predictions and true label
        print("True label = ", y_numpy, "Pred = ",pred_numpy)
        true_labels.extend(y_numpy)
        predicted_labels.extend(pred_numpy)

    #write to a csv file
    image_names.extend([os.path.basename(path) for path, _ in test_loader.dataset.samples])
    csv_path = './predictions.csv'
    df = pd.DataFrame({"Image Name": image_names,"Predicted Label": predicted_labels, "True Label": true_labels}) 
    df.to_csv(csv_path)
    #calculate metrics and log to wandb
    f1 = f1_score(true_labels, predicted_labels,pos_label=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels,pos_label=0)
    recall = recall_score(true_labels, predicted_labels,pos_label=0)
    roc = roc_auc_score(true_labels,predicted_labels)
    wandb.log({"Test_F1 Score": f1})
    wandb.log({"Test_Accuracy": accuracy})
    wandb.log({"Test_Precision": precision})
    wandb.log({"Test_Recall": recall})
    wandb.log({"Roc_Auc": roc})

            


