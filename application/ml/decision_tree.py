import math
import re
import graphviz
from IPython.display import SVG  # TODO change
from IPython.display import display
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz


def modify_tree_to_categorical(name_file):
    """
    This function modifies a file at dot format which is a decision tree. It changes the features names to fit with categorical
    features. It mainly changes the 'oui/non' features.

    It modifies directly the file in the repertory (creates a copy with the expansion "modified") and it returns also
    a string which has a dot format.

    """
    new_file_name = name_file[:-4] + '_modified.dot'
    new_dotfile = open(name_file[:-4] + '_modified.dot', 'w+')
    dotfile = open(name_file, 'r')  # open the dot file where the graph is stored as dot format
    lines = dotfile.readlines()  # we read the lines
    # regex which finds the label and the category of the split
    # for example : ('label="(', "'developpement psychoaffectif'", " 'perturbe'", ')', ' <= 0.5')
    reg = re.compile(r'(.*)(label\=\")\(\'([a-zA-Z _-]+)\'\,\ ([a-zA-Z0-9\(.,=\- \'\])]+)\)([ <=.0-9]+)(.*)')
    # regex for the 1st split explanation 'vrai' 'faux'
    reg2 = re.compile(r'(.*)(headlabel=")([a-zA-Z]+)(.*)')
    reg3 = re.compile(r'(.*)([a-z_]+)\'\,\ ([a-zA-Z]+)\(([0-9.]+)\,\ ([0-9.]+)(.*)')  # regex for the age
    data_dot = ''

    for line in lines:  # the lines of the dot file are covered one by one
        temp_data = ''

        m = re.match(r'(.*)(label\=\")\(\'([a-zA-Z _-]+)\'\,\ ([a-zA-Z0-9\(.,=\-\]\[ \')]+)\)([ <=.0-9]+)(.*)', line)
        m2 = re.match(r'(.*)(headlabel=")([a-zA-Z]+)(.*)', line)
        m3 = re.match(r'(.*)([df_age]+)\'\,\ ([a-zA-Z]+)\(([0-9.]+)\,\ ([0-9.]+)(.*)', line)

        if reg.findall(line):  # there is the pattern in the line
            # group 4 is the category of the feature, for example 'oui' 'non', 'perturbe' etc
            if m.group(4) == "'non'":
                temp_data = reg.sub(r"\1\2Il n'y a pas eu : \3\n\6", line)
            elif m.group(4) == "'oui'":
                temp_data = reg.sub(r"\1\2Il y a eu : \3\n\6", line)
            else:
                val = m.group(4)
                if val[:8] == 'Interval':  # val like : "Interval(66.0, 78.0, closed='right')"
                    # if it concerns the feature 'age', we display it more clearly
                    if m.group(3) == 'df_age' and m3 is not None:
                        age_min = str(math.ceil(float(m3.group(4))))
                        age_max = str(math.floor(float(m3.group(5))))
                        data_age = "l'age vaut entre " + age_min + " et " + age_max + " ans\n"
                        temp_data = m.group(1) + m.group(2) + data_age + m.group(6)

                    if m.group(3) == 'df_sofas':
                        temp_data = reg.sub(r"\1\2le score sofas est dans l'\4\n\6", line)

                elif val[1] == '(' and val[-2] == ']':
                    # print(val,'caarms')
                    temp_data = reg.sub(r"\1\2\3 est dans l'interval \4\n\6", line)

                else:
                    temp_data = reg.sub(r"\1\2\3 vaut \4\n\6", line)
            new_dotfile.write(temp_data)
            data_dot = data_dot + temp_data

        else:  # it s not a line containing information for a branch
            if m2 is not None and m2.group(3) == 'True':
                temp_data = reg2.sub(r"\1\2Vrai\4", line)  # replace 'True' by 'Vrai'
            elif m2 is not None and m2.group(3) == 'False':
                temp_data = reg2.sub(r"\1\2Faux\4", line)  # replace 'False' by 'Faux'
            else:
                temp_data = line
            new_dotfile.write(line)
            data_dot = data_dot + temp_data

    dotfile.close()
    new_dotfile.close()

    return data_dot


def delete_col_decision_tree(col_list, data1, data2):
    """
    This function deletes the columns in the col_list (list of strings) in the training data (data1) and the test data (data2).
    It returns the 2 dataframes without these columns.
    """

    data1_new = data1
    data2_new = data2
    for col in col_list:
        if col in data1.columns:
            data1_new = data1_new.drop([col], axis=1)
            data2_new = data2_new.drop([col], axis=1)

    return data1_new, data2_new


def plot_tree(x_train, y_train, x_test, y_test, classes_name, depth, min_split, min_leaf, weight, display_image=True,
              delete_col=False):
    """
    This function creates an interactive decision tree. This uses the cross validation

    The parameters are :

        - crit is the measurement of the quality of split at each node (either "entropy" : somme sur i de -p(i)*log2(p(i))
        or "gini" : somme p(i)*(1-p(i)) )

        -split The strategy used to choose the split at each node. Supported strategies are “best”
        to choose the best split and “random” to choose the best random split.

        - depth : max depth of the tree, if None, it will go until all leaves are pure or until all leaves
        contain less than min_samples_split samples.

        - min_split :The minimum number of samples required to split an internal node

        - min_leaf :The minimum number of samples required to be at a leaf node

        - weight : Weights associated with classes, dictionary format for the classes 1 (pas de risque), 3(a risque), 5 (psychose)
        If weight = 1, this keeps the number of samples. If weight = 10, this multiplies the original number of samples by 10
        If weight = 'balanced', each class has the same importance

        -display image : bool if true will display the image straight in the cell, if False, it will return the image (used for the app)

    """
    # creates the tree with the parameters of the function
    estimator = DecisionTreeClassifier(random_state=12
                                       , criterion='gini'
                                       , splitter='best'
                                       , max_depth=depth
                                       , min_samples_split=min_split
                                       , min_samples_leaf=min_leaf
                                       , class_weight=weight)

    if not delete_col:  # case there is no deletion of column  (at the beginning of the application)
        print('pas delete')
        # We are going to apply the function delete_col_decision_tree where delete_col is an empty list so nothing is
        # done
        delete_col = [None]
        x_train_filtered, x_test_filtered = delete_col_decision_tree(delete_col, x_train, x_test)  #
    else:
        print('delete', delete_col)
        # if we want to not take a column into consideration for the tree
        x_train_filtered, x_test_filtered = delete_col_decision_tree(delete_col, x_train, x_test)

    print(y_train.unique())
    estimator.fit(x_train_filtered, y_train)  # train the tree on the train data and train label
    # save the graph as 'decision tree.dot' in the current repository
    dot_tree = export_graphviz(estimator
                               , out_file='decision tree.dot'
                               , feature_names=x_train_filtered.columns
                               # link the feature indexes to the feature names (columns of the data frame)
                               , class_names=classes_name
                               # link the label (1,3,5) to the name of each class (pas de risque, a risque, psychose)
                               , filled=True  # paint the node to indicate the majority class
                               , rounded=True)  # graphical style

    # apply the function modify_tree_to_categorical to the tree we created
    dot_tree_modified = modify_tree_to_categorical("decision tree.dot")
    # in order to have a better visual
    graph = graphviz.Source(dot_tree_modified)

    image = graph.pipe(format='png')

    file = open('decision_tree_modified.png', 'wb')
    file.write(image)
    file.close()

    # test of the tree
    prediction = estimator.predict(x_test_filtered)
    confu_matrix = confusion_matrix(y_test, prediction)
    if len(classes_name) == 3:
        classi_report = classification_report(y_test, prediction, [1, 3, 5], classes_name)
    elif len(classes_name) == 2:
        classi_report = classification_report(y_test, prediction, [1, 3], classes_name)
    mutlilab_confu_mat = multilabel_confusion_matrix(y_test, prediction)
    if display_image:
        display(SVG(graph.pipe(format='svg')))
        return None

    else:
        return image, confu_matrix, classi_report, mutlilab_confu_mat,


def best_param_tree(scoring, nb_cv, x_train, y_train):
    """

    :param scoring:
    :param nb_cv:
    :param x_train:
    :param y_train:
    :return:
    """
    param_grid = {'criterion': ['gini'], 'max_depth': [2, 3, 5],
                  'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8]}

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring=scoring, cv=nb_cv,
                               return_train_score=False)

    grid_search.fit(x_train, y_train)

    return grid_search.best_params_, grid_search