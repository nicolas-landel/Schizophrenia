# these two functions are used for numerical variables 'age' and 'sofas score'
# It will create intervals in order to turn these variables into quantitative variables

import pandas as pd
import numpy as np


def creation_disjunctive_array(nb_bins, list_name, list_var):
    """
    This function takes in argument the number of bins you want for all the variables in list_name (list of string of the
    dataframes) and list_var (list of the dataframes)
     It will create a dataframe with 0 or 1 whether the value for each country is within the interval (column name)
    """
    var_index, interval_index = [], []
    disj_df = pd.DataFrame()
    for i, var_name in enumerate(list_name):  # the variables in the list_name are covered

        df_temp = pd.DataFrame()
        var = list_var[i]  # switch name (string) to the dataframe  (I could have done a dictionary)
        df_labelized = labelization(var_name, nb_bins, list_var,
                                    list_name)  # class the values of the variable into the bins, creates a DF
        # array of the repartition of values into the bins and array of the values to cut
        bins_var, delimitation_bins = np.histogram(var, nb_bins)

        # array of the list of intervals, sensitive to high values
        list_intervals = pd.cut(x=var, bins=delimitation_bins, include_lowest=True).unique()
        list_intervals = list_intervals.dropna()  # drop intervals without values
        list_intervals_str = [str(interval) for interval in
                              list_intervals]  # converts intervals to strings for the columns name

        # creation of the columns (multi index) of the future DF for this variable  --> not use properly, to modify
        col_index = pd.MultiIndex.from_arrays([[list_name[i] for p in range(len(list_intervals))], list_intervals],
                                              names=['variable', 'value'])

        var_index.extend(list_name[i] for p in range(len(list_intervals)))
        interval_index.extend(list_intervals_str)  # add the interval as string

        for i, interval in enumerate(list_intervals):
            df_labelized.loc[df_labelized[var_name] == interval, str(col_index[i])] = 1  # multi index not used
            df_labelized.loc[df_labelized[var_name] != interval, str(col_index[i])] = 0

        df_labelized = df_labelized.drop([var_name], axis=1)
        disj_df = pd.concat([disj_df, df_labelized], axis=1)

    global_col_index = pd.MultiIndex.from_arrays([var_index, interval_index],
                                                 names=['variable', 'intervals'])
    disj_df.columns = global_col_index

    return disj_df, global_col_index


def labelization(var_name, nb_bins, list_var, list_name):
    """
    Takes the name of the variable as string format(ex : 'che') and the number of bins
    you want for this variable

    Returns a dataFrame where the values of the df of var_name are categorized into a label (interval)
    """

    # creation empty DataFrame
    df_binned = pd.DataFrame()
    # index of the variable in the lists
    index = list_name.index(var_name)
    df = list_var[index]
    # creation of rows in the dataframe (values set to 0)
    df_binned[var_name] = [i * 0 for i in range(list_var[index].shape[0])]

    bins_var, delimitation_bins = np.histogram(list_var[index], nb_bins)
    df_binned[var_name] = pd.cut(x=df, bins=delimitation_bins, include_lowest=True)

    return df_binned


def creation_disj_df_threshold_split(df, col, lim, name_above, name_below):
    """
    This function create a dataframe from a column of a df where the new df has the original column split
    into 2 new columns. One called 'name_above' with 1 if the value of the row if above the lim, the other called 'name_below'
    if the value of the row is below the threshold
    """

    new_df = pd.DataFrame(data=df[col])

    new_df.loc[new_df[col] >= lim, name_above + ' ' + str(lim)] = 1
    new_df.loc[new_df[col] < lim, name_above + ' ' + str(lim)] = 0

    new_df[name_below + ' ' + str(lim)] = (new_df[name_above + ' ' + str(lim)] - 1) ** 2

    new_df = new_df.drop([col], axis=1)
    new_df.columns = pd.MultiIndex.from_arrays([[col] * 2, [name_above + ' ' + str(lim), name_below + ' ' + str(lim)]]
                                               , names=['features', 'modalities'])
    return new_df


# This cell will ask to the user to define the number of answers possible
# for each column. This will be used for dividing each column into a binary answer


def creation_dic_nb_val_by_col(df):
    """
    This is built to create a dictionary with columns of df_mca as keys and the number of possibles answers as values
    It is very specific and not aims to be reused. Also, there was a previous verion where it was asked for each columns the
    number of possible values. I was very annoying to re-enter this but more scalable.

    It returns the dictionary

    Note that if you want to change the number of possible answers you can either use the manual version of the function
    (function below) or change this function (if the change of possible answers is permanent)
    """
    dic_nb_val = {}
    for col in df.columns:

        if col == "vit avec":
            nb_val = 13
        elif col == "niveau scolaire":
            nb_val = 6
        elif col == "risque sucidaire actuel":
            nb_val = 3
        elif col == "emploi" or col == "addresse par" \
                or col == "traitements psychotropes actuels" \
                or col == "traitements psychotropes passes":
            nb_val = 7
        else:
            nb_val = 2

        dic_nb_val[col] = nb_val
    return dic_nb_val


def creation_dic_nb_val_by_col_label(df):
    """
    Same kind of function than above but adapted to the label where there are 3 modalities by column
    """
    dic_nb_val = {}
    for col in df.columns:
        dic_nb_val[col] = 3
    return dic_nb_val


def creation_manual_dic_nb_val_by_col(df):
    """
    This function has the same purpose that the previous one but the user enters manually the number of possible
    answers for each column of the dataframe
    """
    dic_nb_val = {}
    for col in df.columns:
        nb_val = input()
        dic_nb_val[col] = nb_val


def creation_list_variables_repeted(dic_nb_val):
    """
    This function creates a list of strings which are the variables names repeted the number of possible answers they
    contain (according to dic_nb_val)
    For exapmle, sexe will be repeted 2 times (because two possible answers : male and female)
    """
    var_index = []
    for keys in dic_nb_val:
        var_index = var_index + [keys] * dic_nb_val[keys]
    return var_index


def select_value(df, col, value):
    """
    This function is an intermediary function used to create a temporary new DF which corresponds to a part of the
    column we gonna split
    It takes in argument the dataframe, the name of the column (string) and the value we gonna plit into a new column
    and returns a DF with the same column's name and the binary test (0 or 1) of the presence of the value for each row
    """
    var_index = []
    list_value = []
    for i in range(len(df[col])):
        if value != 0:
            if value == int(df.loc[i, [col]]):
                list_value.append(1)
            else:
                list_value.append(0)
        elif value == 0:
            if int(df.loc[i, [col]]) == 0:
                list_value.append(1)
            else:
                list_value.append(0)
    dataF = pd.DataFrame(np.array(list_value))
    dataF.columns = [col]
    return dataF


def split_columns(df, dic_nb_val, label=False):
    """
    This function will split the columns of the Dataframe DF into the number of possible answers of this column
    The number of possible answers is stored in dic_nb_value (keys = column and value = nb anwers)
    Ex : gender has two possible answers (0 or 1) so the column will be split in two new columns
    This function returns the DF created by these operations
    """
    new_df = pd.DataFrame()
    for col in dic_nb_val.keys():
        nb_col = int(dic_nb_val[col])
        if not label:  # linear incrementation of the value linked to the modality
            # the nb of values possible for each col is covered, in order to split the col into this nb of new col
            for i in range(nb_col):
                new_df = pd.concat((new_df, select_value(df, col, i)),
                                   axis=1)  # pb col est le nom de la colonne et pas la ref au string

        elif label:  # case it s the df_label, the values are 1,3,5 (and not 1,2,3) and 3 modalities
            list_values_feature = sorted(df[col].unique())
            for values in list_values_feature:
                value = int(values)

                new_df = pd.concat((new_df, select_value(df, col, value)),
                                   axis=1)  # pb col est le nom de la colonne et pas la ref au string
    return new_df
