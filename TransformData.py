from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pandas as pd

def get_best_features(data_train, target_train, data_test):
    params_distribs = {
        'n_estimators': randint(low=10, high=25),
        'max_features': randint(low=20, high=50),
    }
    randomForest = RandomForestClassifier(random_state=42)
    rnd_search = RandomizedSearchCV(randomForest, param_distributions=params_distribs, n_iter=50, cv=5,
                                    scoring='accuracy',
                                    random_state=42)

    rnd_search.fit(data_train, target_train)

    rnd_search.predict(data_test)

    feature_importances = rnd_search.best_estimator_.feature_importances_

    dataset_num = pd.DataFrame(data_train)
    num_attribs = list(dataset_num)
    print(len(num_attribs))
    print(sorted(zip(feature_importances, num_attribs), reverse=True))
    return feature_importances


def get_best_dataframe(data_train, data_test, features):
    list_dataset = list(DataFrame(data_train))
    attribs = sorted(zip(features, list_dataset), reverse=True)
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
    return new_dataset_train, new_dataset_test
