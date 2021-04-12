from __future__ import print_function, division
import Classifiers, Plot, TransformData

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
import torchdata as td
class MyDataClass(Dataset):
    def __init__(self, image_path, transform=None):
        super(MyDataClass, self).__init__()
        self.data = datasets.ImageFolder(image_path,  transform) # Create data from folder

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

    def __len__(self):
        return len(self.data)


plt.ion()   # interactive mode
data_dir_all = 'data/ImageListBalanced'

path, dirs, files = next(os.walk(data_dir_all))
total_count = len(files)

# Data augmentation and normalization for training
# Just normalization for validation
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


model_dataset = MyDataClass(data_dir_all, data_transforms)#datasets.ImageFolder(data_dir_all)#torchvision.datasets.ImageFolder(data_dir_all))

# Also you shouldn't use transforms here but below
train_count = int(0.7 * len(model_dataset))
valid_count = int(0.2 * len(model_dataset))
test_count = len(model_dataset) - train_count - valid_count

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    model_dataset, (train_count, valid_count, test_count)
)

print(len(train_dataset))
# Apply transformations here only for train dataset

# Rest of the code goes the same
BATCH_SIZE = 2
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

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
class_names = train_dataset.dataset.data.classes #recupérer classeName

print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

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
                model.eval()  # Set model to evaluate mode

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
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
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
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {} , real : {}'.format(class_names[preds[j]], labels))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def model_to_numpy(model, type):
    label_list = []
    for i in dataloaders[type].dataset.indices:
        label_list.append(dataloaders[type].dataset.dataset.data.targets[i])

    output_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[type]):
            inputs = inputs.to(device)

            outputs = model(inputs)
            ##BatchSize de Dataloaders
            for i in range(len(outputs)):
                output_list.append(outputs[i].cpu().detach().numpy())

    return output_list, label_list


class Model(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.cnn = models.resnet152(pretrained=True)
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
        num_ftrs = self.cnn.fc.in_features

        self.cnn.fc = nn.Linear(num_ftrs, num_ftrs)
        # self.cnn.fc = nn.Linear(num_ftrs, int(num_ftrs/2))
        # self.fc1 = nn.Linear(int(num_ftrs/2), int(num_ftrs/2))
        # self.fc2 = nn.Linear(int(num_ftrs/2), class_num)

    def forward(self, image):
        x = self.cnn(image)

        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x


model_ft = Model(len(class_names))  # models.resnet152(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                       num_epochs=2)

data_train, target_train = model_to_numpy(model_ft, 'train')
data_test, target_test = model_to_numpy(model_ft, 'test')

print(len(data_train), len(target_train))


best_features = TransformData.get_best_features(data_train, target_train, data_test)
clear_dataset_train, clear_dataset_test = TransformData.get_best_dataframe(data_train, data_test, best_features)

all_metric_dict = dict()

scoring_method = ["accuracy", "recall", "roc_auc"]


def scoring_methode(data_train, target_train, data_test, target_test, nb_iteration, cv, scoring_methode_str):
    all_metric_dict.clear()

    all_metric_dict["decision_tree"] = Classifiers.decision_tree(data_train, target_train, data_test, target_test, nb_iteration, cv,
                                                     scoring_methode_str)
    all_metric_dict["RandomForest"] = Classifiers.randomForestClassifierSearch(data_train, target_train, data_test, target_test,
                                                                   nb_iteration, cv, scoring_methode_str)
    all_metric_dict["Logistic_Reg"] = Classifiers.logistic_regression(data_train, target_train, data_test, target_test,
                                                          nb_iteration, cv, scoring_methode_str)
    all_metric_dict["Perceptron"] = Classifiers.function_perceptron(data_train, target_train, data_test, target_test, nb_iteration,
                                                        cv, scoring_methode_str)
    all_metric_dict["gradientBoostingClassifier"] = Classifiers.gradientBoostingClassifier_function(data_train, target_train,
                                                                                        data_test, target_test,
                                                                                        nb_iteration, cv,
                                                                                        scoring_methode_str)
    all_metric_dict["multi_level_Perceptron_Classifier"] = Classifiers.multi_level_Perceptron_Classifier(data_train, target_train,
                                                                                             data_test, target_test,
                                                                                             nb_iteration, cv,
                                                                                             scoring_methode_str)


# scoring_methode(data_train, target_train, data_test, target_test, 10, 5, scoring_method)
scoring_methode(clear_dataset_train, target_train, clear_dataset_test, target_test, 10, 5, scoring_method)

list_dict = Plot.stock_results_dictionnary(all_metric_dict)

Plot.hist_plot([list_dict["test_accuracy_mean_list"], list_dict["test_accuracy_std_list"]], all_metric_dict.keys(),
     ["accuracy_mean", "accuracy_std"], "Accuracy mean/std", 0.2)

Plot.hist_plot([list_dict["test_recall_mean_list"], list_dict["test_recall_std_list"]], all_metric_dict.keys(),
     ["recall_mean", "recall_std"], "Recall mean/std", 0.2)
Plot.hist_plot([list_dict["test_recall_mean_list"], list_dict["test_accuracy_mean_list"], list_dict["test_roc_auc_mean_list"]],
     all_metric_dict.keys(), ["recall", "accuracy", "roc_auc"], "Recall/Accuracy/Roc_auc comparaison", 0.2)

Plot.hist_plot([list_dict["test_recall_mean_list"], list_dict["test_recall_std_list"], list_dict["test_accuracy_mean_list"],
      list_dict["test_accuracy_std_list"], list_dict["test_roc_auc_mean_list"], list_dict["test_roc_auc_std_list"]],
     all_metric_dict.keys(),
     ["recall_mean", "recall_std", "accuracy_mean", "accuracy_std", "roc_auc_mean", "roc_auc_std"],
     "Recall/Accuracy/Roc_auc comparaison", 0.1)

# plot([list_dict["test_recall_mean_list"],list_dict["training_time"]], all_metric_dict.keys(), ["recall_mean","training-time"],"recall_mean/training_time",0.3)


Plot.compare_curve_plot(all_metric_dict,list_dict["training_time"], list_dict["predict_time"], "algorithmes", "training_time", "red", "predict_time","blue")

Plot.compare_curve_plot(all_metric_dict,list_dict["training_time"], list_dict["test_recall_mean_list"], "algorithmes", "training_time", "yellow", "recall_mean", "green")

# model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 2)
#
# model_conv = model_conv.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#
# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=25)

# visualize_model(model_conv)

plt.ioff()
plt.show()
