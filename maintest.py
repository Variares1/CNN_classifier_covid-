from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F

plt.ion()  # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # transforms.Resize(256),
        transforms.CenterCrop((224, 224)),  # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),  # transforms.Resize(256),
        transforms.CenterCrop((224, 224)),  # transforms.CenterCrop(224)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/covid_data_balanced'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

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
    label_list = dataloaders[type].dataset.targets

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
        print(num_ftrs)
        self.cnn.fc = nn.Linear(num_ftrs, num_ftrs)
        # self.cnn.fc = nn.Linear(num_ftrs, int(num_ftrs/2))
        # self.fc1 = nn.Linear(int(num_ftrs/2), int(num_ftrs/2))
        # self.fc2 = nn.Linear(int(num_ftrs/2), class_num)

    def forward(self, image):
        x = self.cnn(image)

        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x


def calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results):
    score_result = {}
    for scorer in scoring_methods:
        for sample in ('train', 'test'):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score_mean = sample_score_mean[best_index]
            best_score_std = sample_score_std[best_index]
            if sample not in score_result.keys():
                score_result[sample] = {}
            score_result[sample][scorer] = {'mean': best_score_mean, 'std': best_score_std}
    print(score_result)
    # print(sample, scorer, best_score_mean)
    # print(sample, scorer, best_score_std)

    return {"training_time": training_time,
            "predict_time": predict_time,
            "train": score_result["train"],
            "test": score_result["test"]
            }


import seaborn as sns
from sklearn import metrics


def matrix_confusion(predictions, labels, score, algo, scoring_methods):
    cm = metrics.confusion_matrix(labels, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    all_sample_title = algo + " : " + scoring_methods + ' Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    # save_fig("Confusion Matrix" + algo)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def decision_tree(data_train, target_train, data_test, target_test, nb_iteration, cv, scoring_methods):
    params_distribs = {
        'max_features': randint(low=5, high=10)
    }

    start_timer = time.perf_counter()
    tree_class = DecisionTreeClassifier(random_state=42)

    rnd_search = RandomizedSearchCV(tree_class, param_distributions=params_distribs, n_iter=nb_iteration, cv=cv,
                                    scoring=scoring_methods, random_state=42, refit='accuracy', return_train_score=True)
    rnd_search.fit(data_train, target_train)
    end_training_time = time.perf_counter()

    prediction = rnd_search.predict(data_test)
    end_predict_time = time.perf_counter()

    training_time = end_training_time - start_timer
    predict_time = end_predict_time - end_training_time

    results = rnd_search.cv_results_

    matrix_confusion(prediction, target_test, rnd_search.best_score_, "DT", scoring_methods[0])

    print(f"Duration Training time: {training_time:0.4f} seconds")
    print(f"Duration Predict time: {predict_time:0.4f} seconds / {scoring_methods[0]} : {rnd_search.best_score_}")
    print(rnd_search.best_estimator_)
    print(rnd_search.best_score_)

    return calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results)


def randomForestClassifierSearch(data_train, target_train, data_test, target_test, nb_iteration, cv, scoring_methods):
    params_distribs = {
        'n_estimators': randint(low=1, high=30),
        'max_features': randint(low=1, high=8),
    }

    start_timer = time.perf_counter()

    randomForest = RandomForestClassifier(random_state=42)
    rnd_search = RandomizedSearchCV(randomForest, param_distributions=params_distribs, n_iter=nb_iteration, cv=cv,
                                    scoring=scoring_methods, random_state=42, refit='accuracy', return_train_score=True)
    rnd_search.fit(data_train, target_train)
    end_training_time = time.perf_counter()

    prediction = rnd_search.predict(data_test)
    end_predict_time = time.perf_counter()

    training_time = end_training_time - start_timer
    predict_time = end_predict_time - end_training_time

    results = rnd_search.cv_results_

    matrix_confusion(prediction, target_test, rnd_search.best_score_, "RFC", scoring_methods[0])

    print(f"Duration Training time: {training_time:0.4f} seconds")
    print(f"Duration Predict time: {predict_time:0.4f} seconds / {scoring_methods[0]} : {rnd_search.best_score_}")
    print(rnd_search.best_estimator_)
    print(rnd_search.best_score_)

    return calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results)


def function_perceptron(data_train, target_train, data_test, target_test, nb_iteration, cv, scoring_methods):
    params_distribs = {
        'max_iter': randint(low=500, high=2000),
        'n_iter_no_change': randint(low=5, high=20),
        # 'tol': random.uniform(1e-2,1e-6),
        # 'shuffle': bool(random.getrandbits(1)),
        # 'penalty': random.choice(['l1','l2','elasticnet']),
    }

    start_timer = time.perf_counter()
    perceptron = Perceptron()

    rnd_search = RandomizedSearchCV(perceptron, param_distributions=params_distribs, n_iter=nb_iteration, cv=cv,
                                    scoring=scoring_methods, random_state=42, refit='accuracy', return_train_score=True)
    rnd_search.fit(data_train, target_train)
    end_training_time = time.perf_counter()

    prediction = rnd_search.predict(data_test)
    end_predict_time = time.perf_counter()

    training_time = end_training_time - start_timer
    predict_time = end_predict_time - end_training_time

    results = rnd_search.cv_results_

    print(f"Duration Training time: {training_time:0.4f} seconds")
    print(f"Duration Predict time: {predict_time:0.4f} seconds / {scoring_methods[0]} : {rnd_search.best_score_}")
    print(rnd_search.best_estimator_)
    print(rnd_search.best_score_)

    matrix_confusion(prediction, target_test, rnd_search.best_score_, "P", scoring_methods[0])

    return calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results)


def logistic_regression(data_train, target_train, data_test, target_test, nb_iteration, cv, scoring_methods):
    params_distribs = {
        'max_iter': randint(low=500, high=1000),
    }

    start_timer = time.perf_counter()
    logisticRegr = LogisticRegression()

    rnd_search = RandomizedSearchCV(logisticRegr, param_distributions=params_distribs, n_iter=nb_iteration, cv=cv,
                                    scoring=scoring_methods, random_state=42, refit='accuracy', return_train_score=True)
    rnd_search.fit(data_train, target_train)
    end_training_time = time.perf_counter()

    prediction = rnd_search.predict(data_test)
    end_predict_time = time.perf_counter()

    results = rnd_search.cv_results_

    training_time = end_training_time - start_timer
    predict_time = end_predict_time - end_training_time

    print(f"Duration Training time: {training_time:0.4f} seconds")
    print(f"Duration Predict time: {predict_time:0.4f} seconds / {scoring_methods[0]} : {rnd_search.best_score_}")
    print(rnd_search.best_estimator_)
    print(rnd_search.best_score_)

    matrix_confusion(prediction, target_test, rnd_search.best_score_, "LR", scoring_methods[0])

    return calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results)


def gradientBoostingClassifier_function(data_train, target_train, data_test, target_test, nb_iteration, cv,
                                        scoring_methods):
    params_distribs = {
        'n_estimators': randint(low=100, high=500),

    }

    start_timer = time.perf_counter()

    gradient_boost_classifier = GradientBoostingClassifier(learning_rate=1.0, max_depth=1)
    rnd_search = RandomizedSearchCV(gradient_boost_classifier, param_distributions=params_distribs, n_iter=nb_iteration,
                                    cv=cv, scoring=scoring_methods, random_state=42, refit='accuracy',
                                    return_train_score=True)
    rnd_search.fit(data_train, target_train)
    end_training_time = time.perf_counter()

    predictions = rnd_search.predict(data_test)
    end_predict_time = time.perf_counter()

    results = rnd_search.cv_results_

    training_time = end_training_time - start_timer
    predict_time = end_predict_time - end_training_time

    matrix_confusion(predictions, target_test, rnd_search.best_score_, "GBC", scoring_methods[0])

    print(f"Duration Training time: {training_time:0.4f} seconds")
    print(f"Duration Predict time: {predict_time:0.4f} seconds / {scoring_methods[0]} : {rnd_search.best_score_}")
    print(rnd_search.best_estimator_)
    print(rnd_search.best_score_)

    return calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results)


def multi_level_Perceptron_Classifier(data_train, target_train, data_test, target_test, nb_iteration, cv,
                                      scoring_methods):
    params_distribs = {
        'max_iter': randint(low=1000, high=2000),

    }

    start_timer = time.perf_counter()
    mlp_classifier = MLPClassifier()
    rnd_search = RandomizedSearchCV(mlp_classifier, param_distributions=params_distribs, n_iter=nb_iteration, cv=cv,
                                    scoring=scoring_methods, random_state=42, refit='accuracy', return_train_score=True)
    rnd_search.fit(data_train, target_train)
    end_training_time = time.perf_counter()

    prediction = rnd_search.predict(data_test)
    end_predict_time = time.perf_counter()

    results = rnd_search.cv_results_

    training_time = end_training_time - start_timer
    predict_time = end_predict_time - end_training_time

    matrix_confusion(prediction, target_test, rnd_search.best_score_, "MLPC", scoring_methods[0])

    print(f"Duration Training time: {training_time:0.4f} seconds")
    print(f"Duration Predict time: {predict_time:0.4f} seconds / {scoring_methods[0]} : {rnd_search.best_score_}")
    print(rnd_search.best_estimator_)
    print(rnd_search.best_score_)

    return calc_and_fill_dictionnary(training_time, predict_time, scoring_methods, results)


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


params_distribs = {
    'n_estimators': randint(low=10, high=25),
    'max_features': randint(low=20, high=50),
}
randomForest = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(randomForest, param_distributions=params_distribs, n_iter=50, cv=5, scoring='accuracy',
                                random_state=42)

rnd_search.fit(data_train, target_train)

rnd_search.predict(data_test)

feature_importances = rnd_search.best_estimator_.feature_importances_

from pandas import DataFrame

dataset_num = pd.DataFrame(data_train)
num_attribs = list(dataset_num)
print(len(num_attribs))
print(sorted(zip(feature_importances, num_attribs), reverse=True))


def get_best_dataframe(data_train,data_test):
    list_dataset = list(DataFrame(data_train))
    attribs = sorted(zip(feature_importances, list_dataset), reverse=True)
    new_list = []
    for i, j in attribs:
        new_list.append(j)
        if i == 0.0:
            print(new_list)
            print(len(new_list))
            break
    old_dataset_train = pd.DataFrame(data_train)
    new_dataset_train = old_dataset_train[new_list]
    old_dataset_test = pd.DataFrame(data_test)
    new_dataset_test = old_dataset_test[new_list]
    return new_dataset_train,new_dataset_test


clear_dataset_train,clear_dataset_test = get_best_dataframe(data_train,data_test)

all_metric_dict = dict()

scoring_method = ["accuracy", "recall", "roc_auc"]


def scoring_methode(data_train, target_train, data_test, target_test, nb_iteration, cv, scoring_methode_str):
    all_metric_dict.clear()

    all_metric_dict["decision_tree"] = decision_tree(data_train, target_train, data_test, target_test, nb_iteration, cv,
                                                     scoring_methode_str)
    all_metric_dict["RandomForest"] = randomForestClassifierSearch(data_train, target_train, data_test, target_test,
                                                                   nb_iteration, cv, scoring_methode_str)
    all_metric_dict["Logistic_Reg"] = logistic_regression(data_train, target_train, data_test, target_test,
                                                          nb_iteration, cv, scoring_methode_str)
    all_metric_dict["Perceptron"] = function_perceptron(data_train, target_train, data_test, target_test, nb_iteration,
                                                        cv, scoring_methode_str)
    all_metric_dict["gradientBoostingClassifier"] = gradientBoostingClassifier_function(data_train, target_train,
                                                                                        data_test, target_test,
                                                                                        nb_iteration, cv,
                                                                                        scoring_methode_str)
    all_metric_dict["multi_level_Perceptron_Classifier"] = multi_level_Perceptron_Classifier(data_train, target_train,
                                                                                             data_test, target_test,
                                                                                             nb_iteration, cv,
                                                                                             scoring_methode_str)


#scoring_methode(data_train, target_train, data_test, target_test, 10, 5, scoring_method)
scoring_methode(clear_dataset_train, target_train, clear_dataset_test, target_test, 10, 5, scoring_method)

def plot(values, labels, metrics, title, bar_width=0.4):
    if type(values[0]) == list:
        bar_list = [plt.bar(range(len(labels)), values[0], width=bar_width)]
        i = 1
        for value in values[1:]:
            bar_list.append(plt.bar([x + bar_width * i for x in range(len(labels))], value, width=bar_width))
            i += 1
        plt.xticks([r + bar_width / len(values) for r in range(len(labels))], labels, rotation=45)
        plt.legend(bar_list, metrics, loc='best', bbox_to_anchor=(1, 0.5))
    else:
        plt.bar(range(len(labels)), values)
        plt.ylabel(metrics[0])
        plt.xticks(range(len(labels)), labels)
    plt.xlabel("algorithme")
    plt.title(title)

    plt.figure(figsize=(8, 6))
    plt.rcParams['figure.dpi'] = 100
    plt.show()


list_dict = {}

for algo in all_metric_dict.keys():
    for sample in ('train', 'test'):
        for score in all_metric_dict[algo][sample].keys():
            for info in ('mean', 'std'):
                key = '%s_%s_%s_list' % (sample, score, info)
                if key not in list_dict.keys():
                    list_dict[key] = []
                list_dict[key].append(all_metric_dict[algo][sample][score][info])
    for value in all_metric_dict[algo]:
        key = '%s' % (value)
        if key not in list_dict.keys():
            list_dict[key] = []
        list_dict[key].append(all_metric_dict[algo][value])

plot([list_dict["test_accuracy_mean_list"], list_dict["test_accuracy_std_list"]], all_metric_dict.keys(),
     ["accuracy_mean", "accuracy_std"], "Accuracy mean/std", 0.2)

plot([list_dict["test_recall_mean_list"], list_dict["test_recall_std_list"]], all_metric_dict.keys(),
     ["recall_mean", "recall_std"], "Recall mean/std", 0.2)
plot([list_dict["test_recall_mean_list"], list_dict["test_accuracy_mean_list"], list_dict["test_roc_auc_mean_list"]],
     all_metric_dict.keys(), ["recall", "accuracy", "roc_auc"], "Recall/Accuracy/Roc_auc comparaison", 0.2)

plot([list_dict["test_recall_mean_list"], list_dict["test_recall_std_list"], list_dict["test_accuracy_mean_list"],
      list_dict["test_accuracy_std_list"], list_dict["test_roc_auc_mean_list"], list_dict["test_roc_auc_std_list"]],
     all_metric_dict.keys(),
     ["recall_mean", "recall_std", "accuracy_mean", "accuracy_std", "roc_auc_mean", "roc_auc_std"],
     "Recall/Accuracy/Roc_auc comparaison", 0.1)

# plot([list_dict["test_recall_mean_list"],list_dict["training_time"]], all_metric_dict.keys(), ["recall_mean","training-time"],"recall_mean/training_time",0.3)

plot1 = plt.plot(all_metric_dict.keys(), list_dict["training_time"], 'ro-', label='training_time')
plt.xticks(rotation=40)
plt.xlabel('algorithmes')
plt.ylabel('training_time', color='red')
plt2 = plt.twinx()
plot2 = plt2.plot(all_metric_dict.keys(), list_dict["predict_time"], 'o-', label='predict_time')
plt2.set_ylabel('predict_time', color='blue')

allplot = plot1 + plot2
alllabels = [l.get_label() for l in allplot]
plt.legend(allplot, alllabels, loc=0)
plt.title('Training_time/Predict_time')
plt.show()

# %%

plot1 = plt.plot(all_metric_dict.keys(), list_dict["training_time"], 'yo-', label='training_time')
plt.xticks(rotation=40)
plt.xlabel('algorithmes')
plt.ylabel('training_time', color='yellow')
plt2 = plt.twinx()
plot2 = plt2.plot(all_metric_dict.keys(), list_dict["test_recall_mean_list"], 'go-', label='recall_mean')
plt2.set_ylabel('recall_mean', color='green')

allplot = plot1 + plot2
alllabels = [l.get_label() for l in allplot]
plt.legend(allplot, alllabels, loc=0)
plt.title('Training_time/Recall_mean')
plt.show()

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
