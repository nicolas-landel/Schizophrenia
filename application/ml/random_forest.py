from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV


def random_forest_grid_search(x_train_rf, y_train_rf, scoring):
    param_grid = {'max_features': [40, 70],
                  'n_estimators': [50, 100],
                  'max_depth': [3, 4],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [2, 5],
                  'max_samples': [0.8],
                  'bootstrap': [True],
                  'oob_score': [True]}

    grid_search = GridSearchCV(RandomForestClassifier(random_state=12), param_grid, scoring=scoring, cv=5,
                               return_train_score=True)

    grid_search.fit(x_train_rf, y_train_rf)

    return grid_search


def creation_random_forest(x_train_rf, y_train_rf, n_estimators, max_depth,
                           min_samples_split, min_samples_leaf, max_features, best_param=False):
    """
    This function creates a random forest a fit it to the data accoring the label

    The parameters are :

        -max_feat = max_features

        -bootstrap

        - depth : max depth of the tree, if None, it will go until all leaves are pure or until all leaves
        contain less than min_samples_split samples.

        - min_split :The minimum number of samples required to split an internal node

        - min_leaf :The minimum number of samples required to be at a leaf node

    """
    if not best_param:
        random_forest = RandomForestClassifier(random_state=12  # creates the tree with the parameters of the function
                                               , max_features=max_features
                                               , max_depth=max_depth
                                               , min_samples_split=min_samples_split
                                               , min_samples_leaf=min_samples_leaf
                                               , bootstrap=True
                                               , oob_score=True
                                               , max_samples=0.9
                                               , n_estimators=n_estimators)
    elif best_param:
        grid_search_ = random_forest_grid_search(x_train_rf, y_train_rf, 'accuracy')
        random_forest = grid_search_.best_estimator_

    random_forest.fit(x_train_rf, y_train_rf)  # train the tree on the train data and train label

    return random_forest


def evaluate_random_forest(x_train_rf, y_train_rf, x_test_rf, y_test_rf, random_forest):
    feature_importances = random_forest.feature_importances_
    list_feature_importance = sorted(zip(feature_importances, x_train_rf.columns), reverse=True)[
                              :10]  # takes the 10 first

    prediction = random_forest.predict(x_test_rf)

    confu_matrix = confusion_matrix(y_test_rf, prediction)
    classi_report = classification_report(y_test_rf, prediction, [1, 3, 5], ['pas de risque', 'a risque', 'psychose'])
    mutlilab_confu_mat = multilabel_confusion_matrix(y_test_rf, prediction)

    return list_feature_importance, confu_matrix, classi_report, mutlilab_confu_mat
