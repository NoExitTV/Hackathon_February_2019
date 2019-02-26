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
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    
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
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
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
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

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
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # Overall accuracy
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Accuracy per class
                c = (preds == labels).squeeze()
                for i in range(10):
                    if i < len(labels):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                    else:
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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Pretrained resnet
        """
        model_ft = models.resnet18(pretrained=False)
        checkpoint = torch.load('./pretrained-models/resnet/model_best.pth.tar')
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}
        model_ft.load_state_dict(state_dict)
        model_ft.eval()
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(64512, num_classes)
        
        # Change last layer
        model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64512, 4096),
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
    
    tobacco_train = datasets.ImageFolder("datasets/Tobacco_split/train",
                                        transform=transform_train)

    tobacco_val = datasets.ImageFolder("datasets/Tobacco_split/val",
                                        transform=transform_val)

    tobacco_test = datasets.ImageFolder("datasets/Tobacco_original/test",
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
    print("Running on device: ", device)
    print("torch.cuda.current_device()", torch.cuda.current_device())
    print("torch.cuda.device(0)", torch.cuda.device(0))
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

batch_size = 16 # Minibatch size
num_epochs = 75
learning_rate = 0.5e-3
num_classes = 10


#%%
########## Run tests ##########

#models_list = ["resnet", "alexnet", "vgg"]
models_list = ["resnet"]
results = []

train_loader, val_loader, test_loader, classes = load_data(0, batch_size)
dataloaders_dict = {"train": train_loader, "test": test_loader, "val": val_loader}

for model_name in models_list:

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
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
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    # Save model
    torch.save(model_ft.state_dict(), "./saved-models/"+model_name+"-rectangular.pth")

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

for m in results:
    print(m['model_name']+":")
    print('Accuracy of the network on the ' + str(m['total']) + ' test images: {:.4f}'.format(100.0 * m['correct'] / m['total']))

    for i in range(10):
        print('Accuracy of {} : {:.4f}'.format(m['classes'][i], 100.0 * m['class_correct'][i] / m['class_total'][i]))        


#%%
########## Plot some stuff ##########
'''
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")

for result in results:
    model_name = result['model_name']
    hist = result['hist']
    ohist = []
    ohist = [h.cpu().numpy() for h in hist]
    plt.plot(range(1,num_epochs+1),ohist,label=model_name)

plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()
'''