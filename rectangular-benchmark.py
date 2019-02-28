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
import os
from shutil import copy as copy2
from shutil import rmtree

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#%%
########## Helper functions ##########

def train_model(model, dataloaders, criterion, optimizer, execution_number, total_runs, num_epochs=25):
    ''' Train the model and then load the weights that gave the best validation results '''

    print("Training on device: {}".format(device))

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        time_elapsed = time.time() - since
        print('Epoch {}/{} [Duration: {:.0f}m {:.0f}s] [Run: {}/{}]'.format(epoch, num_epochs - 1, time_elapsed // 60, time_elapsed % 60, execution_number, total_runs))
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
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print('Best val Acc: {:4f} in Epoch: {:.0f}'.format(best_acc, best_epoch))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} in Epoch: {:.0f}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# def train_model_without_val(model, dataloaders, criterion, optimizer, execution_number, total_runs, num_epochs=25):
#     ''' In this function we do the same training as train_model but we do not load the weights that gave the best validation accuracy.
#         This is the training function that should be ran on the full 100 test images per class (without validation) '''
    
#     print("Training on device: {}".format(device))

#     since = time.time()

#     for epoch in range(num_epochs):
#         time_elapsed = time.time() - since
#         print('Epoch {}/{} [Duration: {:.0f}m {:.0f}s] [Run: {}/{}]'.format(epoch, num_epochs - 1, time_elapsed // 60, time_elapsed % 60, execution_number, total_runs))
#         print('-' * 10)

#         model.train()  # Set model to training mode

#         running_loss = 0.0
#         running_corrects = 0

#         # Iterate over data.
#         for inputs, labels in dataloaders['train']:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward
#             # track history if only in train
#             with torch.set_grad_enabled(True):
                
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 _, preds = torch.max(outputs, 1)

#                 loss.backward()
#                 optimizer.step()   

#             # statistics
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(dataloaders['train'].dataset)
#         epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

#         print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#     return model, [] # Return empty val_acc_history here...

def test_model(model, dataloaders, classes, execution_number, total_runs):
    print("Testing on device: {}".format(device))
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
        print('Testing complete in {:.0f}m {:.0f}s [Run: {}/{}]'.format(time_elapsed // 60, time_elapsed % 60, execution_number, total_runs))
        print('Accuracy of the network on the {} test images: {:.4f} %'.format(total, 100.0 * correct / total))
            
        for i in range(10):
            print('Accuracy of {} : {:.4f} %'.format(classes[i], 100.0 * class_correct[i] / class_total[i]))        
        
        print()

        return correct, total, class_correct, class_total

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model_rectangular(model_name, num_classes, feature_extract, use_pretrained=False):
    ''' Initializes the models that have been pre-trained on rectangular images from imagenet '''

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Pretrained resnet
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        
        # Load pretrained weights
        checkpoint = torch.load('./pretrained-models/resnet/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()

        # Freeze / un-freeze layers
        set_parameter_requires_grad(model_ft, feature_extract)
        
        # Change last fc layer
        model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4096, num_classes)
        )

    elif model_name == "alexnet":
        """ Pretrained alexnet
        """
        model_ft = AlexNet()

        # Load pretrained weights
        checkpoint = torch.load('./pretrained-models/alexnet/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()

        # Freeze / un-freeze layers
        set_parameter_requires_grad(model_ft, feature_extract)
        
        # Change last fc layer
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ Pretrained vgg
        """
        model_ft = vgg13()

        # Load pretrained weights
        checkpoint = torch.load('./pretrained-models/vgg/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()

        # Freeze / un-freeze layers
        set_parameter_requires_grad(model_ft, feature_extract)
        
        # Change last fc layer
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def initialize_model_square(model_name, num_classes, feature_extract, use_pretrained=True):
    ''' Initializes the original models that been trained on square images '''
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        
        # Freeze / un-freeze layers
        set_parameter_requires_grad(model_ft, feature_extract)
        
        # Change last fc layer
        model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(41472, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        
        # Freeze / un-freeze layers
        set_parameter_requires_grad(model_ft, feature_extract)
        
        # Change last fc layer
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        
        # Freeze / un-freeze layers
        set_parameter_requires_grad(model_ft, feature_extract)
        
        # Change last fc layer
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def create_dataset_splits(seed=1337, append_path="NaN"):
    ''' This function creates a dataset split where the randomness depends on the seed value.
        Same seed will generaye the same split. The idea is to increment the seed for every split iteration
        and thus get a new train / test / val split to benchmark the model on.
        The generated dataset will then be saved to disk so that it can be loaded by torchvision dataloaders in another function. '''

    CLASSES = ("ADVE", "Email", "Form", "Letter", "Memo", "News", "Note", "Report", "Resume", "Scientific")
    ROOT = "datasets/Tobacco_test/" + append_path + "/"

    print("Creating new dataset with seed: {} at: {}".format(seed, ROOT))

    # First, remove everything if folders already exist!
    if os.path.exists(ROOT + "test"):
        rmtree(ROOT + "test")
        print("Removed dir {}".format(ROOT + "test"))
    
    if os.path.exists(ROOT + "train"):
        rmtree(ROOT + "train")
        print("Removed dir {}".format(ROOT + "train"))

    if os.path.exists(ROOT + "val"):
        rmtree(ROOT + "val")
        print("Removed dir {}".format(ROOT + "val"))


    def check_and_make_dir(set_name, classes):
        dirs = list(map(lambda x: ROOT + set_name + "/" + x, classes))
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    t = Tobacco("datasets/Tobacco_all/all", num_splits=1, random_state=seed)
    
    # Verify seed
    print("Tobacco random state: {}".format(t.random_state))
    
    phases = ['train', 'val', 'test']
    for phase in phases:
        check_and_make_dir(phase, CLASSES)
        pass

    for phase in phases:
        dir_path = ROOT + phase + "/"
        for i in t.splits[0][phase]:
            dest = dir_path + CLASSES[i[1]] + "/"
            copy2(i[0], dest)
        print("Done copying {} to {}".format(phase, ROOT))
        

def load_data_rectangular(batch_size, append_path=None):
    ''' Loads the images and transform them into rectangular sizes '''

    target_resolution = (480, 640)

    print("Initializing rectangular Datasets and Dataloaders...")
    print("Target resolution: {}".format(target_resolution))

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
    # Load dataset
    #

    # Get path of where to load dataset
    path = "datasets/Tobacco_test/"
    if append_path is not None:
        path = path + append_path + "/"

    print("Loading dataset from path {}".format(path))

    tobacco_train = datasets.ImageFolder(path + "train",
                                        transform=transform_train)

    tobacco_val = datasets.ImageFolder(path + "val",
                                        transform=transform_val)

    tobacco_test = datasets.ImageFolder(path + "test",
                                        transform=transform_test)

    # Load N number of datasets in train dataset
    train_loader = torch.utils.data.DataLoader(dataset=tobacco_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=8)

    # Load n number of datasets into val dataset
    val_loader = torch.utils.data.DataLoader(dataset=tobacco_val,
                                            batch_size=batch_size,
                                            num_workers=8)

    # Load N number of datasets into test dataset
    test_loader = torch.utils.data.DataLoader(dataset=tobacco_test,
                                            batch_size=batch_size,
                                            shuffle=False)

    return train_loader, val_loader, test_loader, tobacco_train.classes   


def load_data_square(batch_size, append_path=None):
    ''' Loads the images and transform them into square sizes '''

    target_resolution = (480, 480)

    print("Initializing square Datasets and Dataloaders...")
    print("Target resolution: {}".format(target_resolution))

    resize_and_crop = torchvision.transforms.Compose([torchvision.transforms.Resize((720, 720)),
                                            torchvision.transforms.RandomCrop(target_resolution)])

    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomChoice([torchvision.transforms.Resize(target_resolution), resize_and_crop]),
                                           transforms.RandomHorizontalFlip(),
                                           torchvision.transforms.RandomRotation((-15,15), resample=False, expand=False, center=None),
                                           torchvision.transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.Resize(target_resolution),
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize(target_resolution),
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # Load dataset
    #

    # Get path of where to load dataset
    path = "datasets/Tobacco_test/"
    if append_path is not None:
        path = path + append_path + "/"

    print("Loading dataset from path {}".format(path))

    tobacco_train = datasets.ImageFolder(path + "train",
                                        transform=transform_train)

    tobacco_val = datasets.ImageFolder(path + "val",
                                        transform=transform_val)

    tobacco_test = datasets.ImageFolder(path + "test",
                                        transform=transform_test)

    # Load N number of datasets in train dataset
    train_loader = torch.utils.data.DataLoader(dataset=tobacco_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=8)

    # Load n number of datasets into val dataset
    val_loader = torch.utils.data.DataLoader(dataset=tobacco_val,
                                            batch_size=batch_size,
                                            num_workers=8)

    # Load N number of datasets into test dataset
    test_loader = torch.utils.data.DataLoader(dataset=tobacco_test,
                                            batch_size=batch_size,
                                            shuffle=False)

    return train_loader, val_loader, test_loader, tobacco_train.classes  


#%%
########## Setup parameters ##########

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Running on device: {}".format(device))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    print("torch.cuda.device(0): {}".format(torch.cuda.device(0)))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print("torch.cuda.get_device_name(0): {}".format(torch.cuda.get_device_name(0)))

batch_size = 16 # Minibatch size
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-1
num_classes = 10
number_of_different_splits = 1


#%%
########## Run tests ##########

# models_list = ["resnet" for i in range(5)]  # Run the same model 5 times and calc average
models_list = ["resnet"]
results = []
t0 = time.time()

#
# First we iterate over i which is on how many splits we want to test or model on
# We then run all models present in 'models_list'. To run the same model many times for each split,
# just add the same model name multiple times into the 'models_list'
#

for split_num in range(number_of_different_splits):
    
    # Create new dataset (different seed) and save it to disk
    create_dataset_splits(seed=1337+split_num, append_path=str(split_num))

    # Initialize data loaders and save in dict
    train_loader, val_loader, test_loader, classes = load_data_square(batch_size, append_path=str(split_num)) # Square
    #train_loader, val_loader, test_loader, classes = load_data_rectangular(batch_size, append_path=str(split_num)) # Rectangular
    
    dataloaders_dict = {"train": train_loader, "test": test_loader, "val": val_loader}

    model_num = 0 # Keep track of the number of runs we've been doing
    for model_name in models_list:

        total_runs = number_of_different_splits * len(models_list)
        model_num += 1

        # Keep track of this so we can plot stuff
        execution_number = (split_num + 1) * model_num

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = False

        # Initialize the model for this run
        model_ft, input_size = initialize_model_square(model_name, num_classes, feature_extract, use_pretrained=True) # Square
        # model_ft, input_size = initialize_model_rectangular(model_name, num_classes, feature_extract, use_pretrained=False) # Rectangular

        # Print the model we just instantiated
        if execution_number == 1:
            print(model_ft)

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
        print("Using learning rate {} and weight decay {}".format(learning_rate, weight_decay))
        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, execution_number, total_runs, num_epochs=num_epochs)

        # Save model
        torch.save(model_ft.state_dict(), "./saved-models/" + model_name + "-square-" + str(execution_number) + ".pth")

        # Test
        correct, total, class_correct, class_total = test_model(model_ft, dataloaders_dict, classes, execution_number, total_runs)

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
print()
print("Testing done! Calculating results...")
for m in results:

    total += m['total']
    total_correct += m['correct']

    print(m['model_name']+":")
    print('Accuracy of the network on the {} test images: {:.4f} %'.format(m['total'], 100.0 * m['correct'] / m['total']))

    for i in range(10):
        print('Accuracy of {} : {:.4f} %'.format(m['classes'][i], 100.0 * m['class_correct'][i] / m['class_total'][i]))        
    
    print()

# Print average accuracy
time_elapsed = time.time() - t0
print("\n")
print('Average accuracy on {} test images in {} number of runs: {:.4f} %'.format(total, number_of_different_splits * len(models_list), 100.0 * total_correct / total))
print("Total correct: {}, Total number of images: {}".format(total_correct, total))
print('Total runtime: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))