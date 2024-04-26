import torch
from torch.utils.data import  DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# because the original test set doesnt includes label, it is hard to use imagefolder, so we write our own class
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    




def predict(model,myTransform,config,device):

    # test dataset
    Screw_test_path = config['Screw_Img_path']+"/archive/test"
    test_dataset = TestImageDataset(folder_path= Screw_test_path, transform=myTransform)
    test_loader = DataLoader(test_dataset)

    state_dict = torch.load('./models/model_best.ckpt')
    model.load_state_dict(state_dict)
    print("TEstmodel : ",model)
    model.eval()

    for x in tqdm(test_loader):
        x = x.to(device)
        print(x.shape)
        with torch.no_grad():
            pred = model(x)
            # if the tensor output is <=0.5, it means the screw is undamaged/good
            Screw_class = "good" if pred <=0.5 else "bad"
            print( "Prediction = ", pred, "class=",Screw_class)
            np_image = x[0].cpu().numpy().transpose((1, 2, 0))
            plt.imshow(np_image)
            plt.title(f'Predicted: {Screw_class} ')
            plt.show()


