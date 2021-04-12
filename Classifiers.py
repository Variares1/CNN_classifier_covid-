from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import numpy as np

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


def matrix_confusion(predictions, labels, score, algo, scoring_methods):
    cm = metrics.confusion_matrix(labels, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    all_sample_title = algo + " : " + scoring_methods + ' Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    # save_fig("Confusion Matrix" + algo)


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
