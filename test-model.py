#%% 
########## Imports ##########
import torch as torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from alexnet import AlexNet
from vgg import VGG, vgg13

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#%%
########## Helper functions ##########
def test_model(model, dataloaders, classes):
    print("Testing on device: ", device)
    with torch.no_grad():
        since = time.time()

        # Vars for total accuracy
        correct = 0
        total = 0
        # Vars for accuracy per class
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for inputs, labels in dataloaders['test']:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Overall accuracy
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Accuracy per class
                c = (preds == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                    if not(i < len(labels)):
                        print("### WIERD ERROR ###")
                        print("i: ", i)
                        print("len(labels): ", len(labels))
                        print("labels: ", labels)

        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Accuracy of the network on the ' + str(total) + ' test images: {:.4f}'.format(100.0 * correct / total))
            
        for i in range(10):
            print('Accuracy of {} : {:.4f}'.format(classes[i], 100.0 * class_correct[i] / class_total[i]))        
        
        return correct, total, class_correct, class_total

def initialize_model(model_name, num_classes, feature_extract, saved_model_path):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Pretrained resnet
        """
        model_ft = models.resnet18(pretrained=False)
        model_ft.load_state_dict(saved_model_path)
        model_ft.eval()

        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(64512, num_classes)
        
        # Change last layer
        # model_ft.fc = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(64512, 4096),
        #     nn.Linear(4096, num_classes)
        # ) 

    elif model_name == "alexnet":
        """ Pretrained alexnet
        """
        model_ft = AlexNet()
        checkpoint = torch.load('./pretrained-models/alexnet/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ Pretrained vgg
        """
        model_ft = vgg13()
        checkpoint = torch.load('./pretrained-models/vgg/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def load_data(input_size, batch_size):
    target_resolution = (480, 640)

    print("Initializing Datasets and Dataloaders...")
    
    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize(target_resolution),
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tobacco_test = datasets.ImageFolder("datasets/Tobacco_split/test",
                                        transform=transform_test)

    # Load N number of datasets into test dataset
    test_loader = torch.utils.data.DataLoader(dataset=tobacco_test,
                                            batch_size=batch_size,
                                            shuffle=False)

    return test_loader, tobacco_test.classes    


#%%
########## Setup parameters ##########

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Running on device: ", device)
    print("torch.cuda.current_device()", torch.cuda.current_device())
    print("torch.cuda.device(0)", torch.cuda.device(0))
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

num_classes = 10
best_resnet_model_path = "./saved-models/resnet-rectangular-80images-best.pth"

#%%
########## Run tests ##########

#models_list = ["resnet", "alexnet", "vgg"]
models_list = ["resnet"]
results = []

test_loader, classes = load_data(0, 0)
dataloaders_dict = {"train": None, "test": test_loader, "val": None}

for model_name in models_list:

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, best_resnet_model_path)

    # Print the model we just instantiated
    print(model_ft)

    # Send the model to device (hopefully GPU :))
    model_ft = model_ft.to(device)

    # Load model
    model_ft.load_state_dict()
    model_ft.eval()

    # Test
    correct, total, class_correct, class_total = test_model(model_ft, dataloaders_dict, classes)

    # Add model results
    results.append({'model_name': model_name,
                    'correct': correct,
                    'total': total,
                    'class_correct': class_correct,
                    'class_total': class_total,
                    'classes': classes})

    # Some memory management!
    del model_ft
    torch.cuda.empty_cache()

for m in results:
    print(m['model_name']+":")
    print('Accuracy of the network on the ' + str(m['total']) + ' test images: {:.4f}'.format(100.0 * m['correct'] / m['total']))

    for i in range(10):
        print('Accuracy of {} : {:.4f}'.format(m['classes'][i], 100.0 * m['class_correct'][i] / m['class_total'][i]))        