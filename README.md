## This code is used to train a classification model to detect anomaly screws in a given MVTec Screws dataset.

# How to run the code

## Libraries

- Directly running the main.py will download the requirement libraries, after finishing downloading, you might face the option to login to W&B account, you can either choose not to visualize the metrics and proceed with training, testing..., or follow the instructions to create or login to an existing W&B account to visualize the metrics.

## Log into Wandb

- This code utilize Wandb to log and plot all the metrics, you will need to login to Wandb in order to visualize the metrics
- You can follow the guide from the terminal or access to this page (https://docs.wandb.ai/quickstart) to login into Wandb

## How to train and test the model

- There are three main functions in this code: 
    1.training
    2.testing with testing dataset that I manually labelled, will calculate the Roc_auc,accuracy, F1 score..., will also output the result as a csv file.
    3.testing with no true label, every test image will pop up one by one with the prediction as the image title. 

- The code accepts argument parsing of three functions, corresponding to the above main functions  :
    1.train  
    2.predict_w_class (Recommended to use this rather than 3. for overall prediction)
    3.predict   (Use to inspect each output one by one, will take some time to go through every image)

- type `python main.py` to run through all three functions in the order of 1,2,3

- or you can specific one or more function you want to use, for example we only want to execute the functions in the order of 1,3,2: 
  `python main.py --function train predict predict_w_class`

- or only execute the 1.&2. function (Recommended):
  `python main.py --function train predict_w_class`


# Some parameters about the code

## Config

- Inside the main.py, you can find the config block including seed, epochs and other parameters. The current config is the exact same as the model I trained, so it should reproduce the same result as my presentation. You can also modify yourself for other testing, just in case I put the original config here for backup.

```    
config ={
        'seed': 1234,      
        'train_ratio': 0.8,   
        'epochs': 500,    
        'batch_size': 16,
        'learning_rate': 1e-4,
        'early_stop': 10,    
        'save_path': './models/model.ckpt', 
        'Screw_Img_path' : "./Screw_Img"
    }
```

## Normalization 

- The normalization I used this time is shown as below, I find the result to be better when the contrast of image is higher, feel free to modify it
```
transforms.Normalize(            
            mean=[0.2, 0.2, 0.2],
            std=[0.7, 0.7, 0.7])
```

## Pretrained model

- For this model, I choose ResNet18 as the pretrained model, and add a fully connected layer of 512 input features and 1 output feature to match our binary classification usage. I then choose sigmoid function to clamp the logits into the range of `[0,1]`, and also to match the BCE loss function that we used.

