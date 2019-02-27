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
from dataset import Tobacco

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#%%
########## Helper functions ##########
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    ''' Train the model and then load the weights that gave the best validation results '''

    print("Training on device: ", device)

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def train_model_without_val(model, dataloaders, criterion, optimizer, num_epochs=25):
    ''' In this function we do the same training as train_model but we do not load the weights that gave the best validation accuracy.
        This is the training function that should be ran on the full 100 test images per class (without validation) '''
    
    print("Training on device: ", device)

    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()   

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, [] # Return empty val_acc_history here...

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
                    

        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Accuracy of the network on the ' + str(total) + ' test images: {:.4f} %'.format(100.0 * correct / total))
            
        for i in range(10):
            print('Accuracy of {} : {:.4f} %'.format(classes[i], 100.0 * class_correct[i] / class_total[i]))        
        
        return correct, total, class_correct, class_total

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Pretrained resnet
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        checkpoint = torch.load('./pretrained-models/resnet/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(64512, num_classes)
        
        # Change last fc layer
        model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        ) 

    elif model_name == "alexnet":
        """ Pretrained alexnet
        """
        model_ft = AlexNet()
        checkpoint = torch.load('./pretrained-models/alexnet/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()
        set_parameter_requires_grad(model_ft, feature_extract)
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
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def load_data(input_size, batch_size):
    target_resolution = (480, 640)

    print("Initializing Datasets and Dataloaders...")

    resize_and_crop = torchvision.transforms.Compose([torchvision.transforms.Resize((720, 960)),
                                            torchvision.transforms.RandomCrop(target_resolution)])

    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomChoice([torchvision.transforms.Resize(target_resolution), resize_and_crop]),
                                           transforms.RandomHorizontalFlip(),
                                           torchvision.transforms.RandomRotation((-15,15), resample=False, expand=False, center=None),
                                           #torchvision.transforms.RandomVerticalFlip(),
                                           torchvision.transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize(target_resolution),
    #                                         transforms.RandomHorizontalFlip(),
    #                                         torchvision.transforms.ToTensor(),
    #                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.Resize(target_resolution),
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize(target_resolution),
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # Load static dataset
    #
    ''' Uncomment the below lines to load a static dataset saved on disk '''
    tobacco_train = datasets.ImageFolder("datasets/Tobacco_split/train",
                                        transform=transform_train)
    # tobacco_val = datasets.ImageFolder("datasets/Tobacco_split/val",
    #                                     transform=transform_val)

    # tobacco_test = datasets.ImageFolder("datasets/Tobacco_split/test",
    #                                     transform=transform_test)

    # # Load N number of datasets in train dataset
    # train_loader = torch.utils.data.DataLoader(dataset=tobacco_train,
    #                                         batch_size=batch_size,
    #                                         shuffle=True,
    #                                         num_workers=8)

    # # Load n number of datasets into val dataset
    # val_loader = torch.utils.data.DataLoader(dataset=tobacco_val,
    #                                         batch_size=batch_size,
    #                                         num_workers=8)

    # # Load N number of datasets into test dataset
    # test_loader = torch.utils.data.DataLoader(dataset=tobacco_test,
    #                                         batch_size=batch_size,
    #                                         shuffle=False)

    # classes = tobacco_train.classes


    #
    # use Tobacco and create a random split!
    #
    ''' Use this only if you want to load the entire dataset and create a random train/val/test split '''
    
    tobacco_train = Tobacco("datasets/Tobacco_all/all",
                            transform=transform_train)
    tobacco_val = Tobacco("datasets/Tobacco_all/all",
                            transform=transform_val)
    tobacco_test = Tobacco("datasets/Tobacco_all/all", 
                            transform=transform_test)
    
    # Load N number of datasets in train dataset
    tobacco_train.load_split("train")
    print("Tobacco train len: ", len(tobacco_train))
    train_loader = torch.utils.data.DataLoader(dataset=tobacco_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=8)

    # Load n number of datasets into val dataset
    tobacco_val.load_split("val")
    print("Tobacco val len: ", len(tobacco_val))
    val_loader = torch.utils.data.DataLoader(dataset=tobacco_val,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8)

    # Load N number of datasets into test dataset
    tobacco_test.load_split("test")
    print("Tobacco test len: ", len(tobacco_test))
    test_loader = torch.utils.data.DataLoader(dataset=tobacco_test,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    # DEBUG
    print(train_loader)
    print(len(train_loader))
    i = 0
    for inputs, labels in train_loader:
        i += 1
        print(inputs.shape)
        print(labels.shape)
        if i > 10:
            break


    return train_loader, val_loader, test_loader, tobacco_train.classes   


#%%
########## Setup parameters ##########

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Running on device: ", device)
    print("torch.cuda.current_device()", torch.cuda.current_device())
    print("torch.cuda.device(0)", torch.cuda.device(0))
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

batch_size = 16 # Minibatch size
num_epochs = 100
learning_rate = 0.5e-3
num_classes = 10
number_of_different_splits = 3

#%%
########## Run tests ##########

# models_list = ["resnet" for i in range(5)]  # Run the same model 5 times and calc average
models_list = ["resnet"]
results = []

#
# First we iterate over i which is on how many splits we want to test or model on
# We then run all models present in 'models_list'. To run the same model many times for each split,
# just add the same model name multiple times into the 'models_list'
#

for i in range(number_of_different_splits):
    
    train_loader, val_loader, test_loader, classes = load_data(0, batch_size)
    dataloaders_dict = {"train": train_loader, "test": test_loader, "val": val_loader}

    run_numb = 1 # Keep track of the number of runs we've been doing
    for model_name in models_list:

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = False

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

        # Print the model we just instantiated
        if run_numb == 1:
            print(model_ft)
            
        print("\nRun number: {} / {}\n".format(run_numb, len(models_list)))
        run_numb += 1

        # Send the model to device (hopefully GPU :))
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate) # We could try and implement a degrading learning rate!

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        # Save model
        torch.save(model_ft.state_dict(), "./saved-models/" + model_name + "-rectangular.pth")

        # Test
        correct, total, class_correct, class_total = test_model(model_ft, dataloaders_dict, classes)

        # Add model results
        results.append({'model_name': model_name, 
                        'hist': hist, 
                        'correct': correct,
                        'total': total,
                        'class_correct': class_correct,
                        'class_total': class_total,
                        'classes': classes})

        # Some memory management!
        del model_ft
        torch.cuda.empty_cache()

    # More memory management
    del dataloaders_dict
    del train_loader
    del val_loader
    del test_loader
    del classes

# Print results and calculate average
total = 0
total_correct = 0
for m in results:

    total += m['total']
    total_correct += m['correct']

    print(m['model_name']+":")
    print('Accuracy of the network on the ' + str(m['total']) + ' test images: {:.4f} %'.format(100.0 * m['correct'] / m['total']))

    for i in range(10):
        print('Accuracy of {} : {:.4f} %'.format(m['classes'][i], 100.0 * m['class_correct'][i] / m['class_total'][i]))        

# Print average accuracy
print("\n")
print('Average accuracy on ' + str(total) + ' test images in ' +  str(len(models_list)) + ' number of runs: {:.4f} %'.format(100.0 * total_correct / total))
print("Total correct: {} | Total number of images: {}".format(total_correct, total))