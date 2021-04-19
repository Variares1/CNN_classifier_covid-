from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

plt.ion()   # interactive mode


class MyDataClass(Dataset):
    def __init__(self, image_path, transform=None):
        super(MyDataClass, self).__init__()
        self.data = datasets.ImageFolder(image_path,  transform) # Create data from folder

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.cnn = models.resnet152(pretrained=True)
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, int(num_ftrs/2))
        self.fc1 = nn.Linear(int(num_ftrs/2), int(num_ftrs/2))
        self.fc2 = nn.Linear(int(num_ftrs/2), class_num)

    def forward(self, image):
        x = self.cnn(image)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_sk = 0.0
    best_recal_sk = 0.0
    best_roc_auc_sk = 0.0

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
            running_preds = []
            running_labels = []

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
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                [running_preds.append(x.cpu().numpy()) for x in preds.data]
                [running_labels.append(x.cpu().numpy()) for x in labels.data]
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc_sk = precision_score(running_labels, running_preds)
            epoch_recal_sk = recall_score(running_labels, running_preds)
            epoch_roc_auc_sk = roc_auc_score(running_labels, running_preds)
            print('{} Loss: {:.4f} Acc: {:.4f} Acc sk:{:.4f} Recal sk:{:.4f} Roc Auc sk:{:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_acc_sk, epoch_recal_sk, epoch_roc_auc_sk))

            # deep copy the model
            if phase == 'val' and epoch_acc_sk > best_acc_sk:
                best_acc = epoch_acc
                best_acc_sk = epoch_acc_sk
                best_recal_sk = epoch_recal_sk
                best_roc_auc_sk = epoch_roc_auc_sk
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} Acc sk:{:4f} Recal sk:{:4f} Roc Auc sk:{:4f}'.format(best_acc, best_acc_sk,
                                                                                       best_recal_sk, best_roc_auc_sk))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "./model_{:4f}.pth".format(best_acc_sk))
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def test_model(model):
    was_training = model.training
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_preds = []
    running_labels = []
    pred_time = 0.0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            start = time.time()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            end = time.time()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            [running_preds.append(x.cpu().numpy()) for x in preds.data]
            [running_labels.append(x.cpu().numpy()) for x in labels.data]
            pred_time = end - start

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    epoch_acc_sk = precision_score(running_labels, running_preds)
    epoch_recal_sk = recall_score(running_labels, running_preds)
    epoch_roc_auc_sk = roc_auc_score(running_labels, running_preds)
    epoch_matrix = confusion_matrix(running_labels, running_preds)
    print('{} Loss: {:.4f} Acc: {:.4f} Acc sk:{:.4f} Recal sk:{:.4f} Roc Auc sk:{:.4f} Pred time:{:.4f}'.format(
        'test', epoch_loss, epoch_acc, epoch_acc_sk, epoch_recal_sk, epoch_roc_auc_sk, pred_time))
    print(epoch_matrix)
    model.train(mode=was_training)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((224, 224)),#transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((224, 224)),#transforms.Resize(256),
#         transforms.CenterCrop((224, 224)), #transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize((224, 224)),  # transforms.Resize(256),
#         transforms.CenterCrop((224, 224)),  # transforms.CenterCrop(224)
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# }

data_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

data_dir = 'data/ImageList'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val', 'test']}
model_dataset = MyDataClass(data_dir, data_transforms)
train_count = int(0.7 * len(model_dataset))
valid_count = int(0.2 * len(model_dataset))
test_count = len(model_dataset) - train_count - valid_count

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    model_dataset, (train_count, valid_count, test_count)
)
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
#                                              shuffle=True, num_workers=0)
#               for x in ['train', 'val', 'test']}
BATCH_SIZE = 64
NUM_WORKER = 0
train_dataset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
valid_dataset_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
test_dataset_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER
)
dataloaders = {
    "train": train_dataset_loader,
    "val": valid_dataset_loader,
    "test": test_dataset_loader
}
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}
class_names = train_dataset.dataset.data.classes#image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

model_ft = Model(len(class_names))#models.resnet152(pretrained=True)
#num_ftrs = model_ft.fc.in_features
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
visualize_model(model_ft)

test_model(model_ft)

plt.ioff()
plt.show()