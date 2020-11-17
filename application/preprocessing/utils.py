from copy import deepcopy

import pandas as pd


def lower_text_column(col, df):
    """
    This function lowers the text values of a column of a dataframe
    It takes in argument a column name (string) and a dataframe
    It returns the dataframe with the text values of the dataframe lowered
    If there is no text value (Nan), it replaces them by empty string
    """
    df[col] = df.loc[:, col].fillna(value='')
    for i in range(len(df[col])):
        df.loc[i, col] = df.loc[i, col].lower()
    return df


def word_in_column(word, column, new_name_column, df):
    """
    This function takes in argument a word (string), the name of a column in the dataframe, the name of the
    new column which will be created in the dataFrame and the dataframe (DF)
    It will create a new column in the DF with 0 (no) or 1 (yes), depending on if the word is in the column of the DF
    originally or not
    """
    df[new_name_column] = df[column].apply(lambda x: 1 if word in x else 0)  # 1 is the word is in the column, 0 if not


def creation_col_caarms(df, period):
    """
    This functions selects the features for the score caarms and calculates its value
    """
    df_bool = df.isna()
    df_caarms_calculated = pd.DataFrame(index=df.index, columns=["score caarms"])
    for i in df.index:  # select the last evalution of caarms values for each patient
        if period == 0:
            list_col_caarms = ['trouble', 'q74', 'q75', 'q76', 'q77', 'q78', 'q79', 'q80']

        elif period == 1:
            # There is an evaluation of the score caarms for the patient i at the 2nd consultation
            if not df_bool.loc[i, 'trouble1']:
                list_col_caarms = ['trouble1', 'q82', 'q83', 'q84', 'q85', 'q86', 'q87', 'q88']
            else:
                list_col_caarms = ['trouble', 'q74', 'q75', 'q76', 'q77', 'q78', 'q79', 'q80']

        elif period == 2:
            if not df_bool.loc[i, 'trouble2']:
                list_col_caarms = ['trouble2', 'q90', 'q91', 'q92', 'q93', 'q94', 'q95', 'q96']
            elif df_bool.loc[i, 'trouble2']:  # no evaluation, we need to look to the formers period
                if not df_bool.loc[i, 'trouble1']:
                    list_col_caarms = ['trouble1', 'q82', 'q83', 'q84', 'q85', 'q86', 'q87', 'q88']
                else:
                    list_col_caarms = ['trouble', 'q74', 'q75', 'q76', 'q77', 'q78', 'q79', 'q80']

        elif period == 3:
            if not df_bool.loc[i, 'trouble3']:
                list_col_caarms = ['trouble3', 'q98', 'q99', 'q100', 'q101', 'q102', 'q103', 'q104']
            else:
                if not df_bool.loc[i, 'trouble2']:
                    list_col_caarms = ['trouble2', 'q90', 'q91', 'q92', 'q93', 'q94', 'q95', 'q96']
                else:
                    if not df_bool.loc[i, 'trouble1']:
                        list_col_caarms = ['trouble1', 'q82', 'q83', 'q84', 'q85', 'q86', 'q87', 'q88']
                    else:
                        list_col_caarms = ['trouble', 'q74', 'q75', 'q76', 'q77', 'q78', 'q79', 'q80']

        elif period == 4:
            if not df_bool.loc[i, 'trouble4']:
                list_col_caarms = ['trouble4', 'q106', 'q107', 'q108', 'q109', 'q110', 'q111', 'q112']
            else:
                if not df_bool.loc[i, 'trouble3']:
                    list_col_caarms = ['trouble3', 'q98', 'q99', 'q100', 'q101', 'q102', 'q103', 'q104']
                else:
                    if not df_bool.loc[i, 'trouble2']:
                        list_col_caarms = ['trouble2', 'q90', 'q91', 'q92', 'q93', 'q94', 'q95', 'q96']
                    else:
                        if not df_bool.loc[i, 'trouble1']:
                            list_col_caarms = ['trouble1', 'q82', 'q83', 'q84', 'q85', 'q86', 'q87', 'q88']
                        else:
                            list_col_caarms = ['trouble', 'q74', 'q75', 'q76', 'q77', 'q78', 'q79', 'q80']

        df_caarms = df.loc[i, list_col_caarms]
        df_caarms_calculated.iloc[i, :] = df_caarms.iloc[0] * df_caarms.iloc[1] + \
                                          df_caarms.iloc[2] * df_caarms.iloc[3] + \
                                          df_caarms.iloc[4] * df_caarms.iloc[5] + \
                                          df_caarms.iloc[6] * df_caarms.iloc[7]
    return pd.DataFrame(df_caarms_calculated)


def select_label(df, col="q80bis", drop_lost=True):
    """
    This function gathers the label 'a risque' and drop the patient 'perdu de vue'
    """
    match_patient = {}  # dic to link the index of a dataFrame to a patient in the excel file.
    list_indexes_dropped = []
    j = 0  # index
    for i in df.index:
        if df.loc[i, col] == 4 or df.loc[i, col] == 2:
            df.loc[i, col] = 3
            match_patient[j] = i
            j += 1
        elif df.loc[i, col] == 6 or df.loc[i, col] == 7:
            if drop_lost:
                list_indexes_dropped.append(i)
                df = df.drop([i], axis=0)  # patient 'perdus de vue' are dropped of the study
            else:
                df.loc[i, col] = 7
        else:
            match_patient[j] = i
            j += 1

    return df, match_patient, list_indexes_dropped


def emptyVal(df):
    """
    This function takes the DF in argument and will return :
    Count (int) the numer of empty value
    dic_empty (dic) the dictionary with the columns of DF as the keys and the list of the lines where there is an empty val
    as values
    """
    count = 0
    dic_empty_position = {}
    Na_df = pd.isna(df)
    for i in Na_df.columns:
        dic_empty_position[i] = []
        L = []
        for j in range(0, len(Na_df[i])):
            if Na_df.loc[j, [i]].bool():
                L.append(j)
                dic_empty_position[i] = L
                count = count + 1
    return count, dic_empty_position


def fill_Na_most_commun(df, column):
    """this function fill the column with the most common value of the column"""
    common_value = df[column].value_counts().idxmax()
    df[column] = df[column].fillna(common_value)
    return df, common_value


def fill_Na_value(df, column, value):
    df[column] = df[column].fillna(value)
    return df, value


def select_empty_value_dic(dic):
    """ this function selects the columns where there isn t any empty value en the DF"""
    list_keys = []
    for i in dic.keys():
        if len(dic[i]) == 0:
            list_keys.append(i)
    return list_keys


def creation_dic_matching_names(df, list_new_names):
    """ This function takes in argument a dataframe, usually df_selected_filled_ready where the columns are not renamed yet
    and the list of the new names (list of strings) we want to assign to the df.
    Note that the length of the two elements must have the same length

    It returns a dictionary where the keys are the df columns names and the values the futur df columns names

    It is used to be sure that the match is good between the col and the names"""

    dic = {}
    try:
        for i, columns in enumerate(df.columns):
            dic[columns] = list_new_names[i]
    except:
        # todo remove bare except
        print('the dataframe and the list have not the same lenght')
    return dic


def add_binary_column_to_df(df, list_columns, name_new_col):
    """
    This function is used to create the new variables 'dependance' and 'traumatisme'

    It takes in argument a dataframe, a list of columns (list of strings) and the name of the new column (string)

    It returns the same df with an additional column named as defined. This column is composed of 0 and 1 whether the
    columns in the list of columns were 0 or 1. If there is one 1 or more in these columns, it will be a 1.
     If there are only 0, it will be 0.

    """
    nb_col = len(list_columns)
    if nb_col == 2:
        df.loc[(df[list_columns[0]] == 1) & (df[list_columns[1]] == 1), name_new_col] = int(1)
        df.loc[(df[list_columns[0]] == 0) & (df[list_columns[1]] == 0), name_new_col] = int(0)

    elif nb_col == 3:
        df.loc[
            (df[list_columns[0]] == 1) | (df[list_columns[1]] == 1) | (df[list_columns[2]] == 1), name_new_col] = int(
            1)  # if one value of the columns is 'yes', the value is set to yes
        df.loc[
            (df[list_columns[0]] == 0) & (df[list_columns[1]] == 0) & (df[list_columns[2]] == 0), name_new_col] = int(
            0)  # else, all are 'no' so the value is set to 'no'

    elif nb_col == 4:
        df.loc[(df[list_columns[0]] == 1) | (df[list_columns[1]] == 1) | (df[list_columns[2]] == 1) | (
                df[list_columns[3]] == 1), name_new_col] = int(
            1)  # if one value of the columns is 'yes', the value is set to yes
        df.loc[(df[list_columns[0]] == 0) & (df[list_columns[1]] == 0) & (df[list_columns[2]] == 0) & (
                df[list_columns[3]] == 0), name_new_col] = int(0)  # else, all are 'no' so the value is set to 'no'

    elif nb_col == 5:
        df.loc[(df[list_columns[0]] == 1) | (df[list_columns[1]] == 1) |
               (df[list_columns[2]] == 1) | (df[list_columns[3]] == 1) | (
                       df[list_columns[4]] == 1), name_new_col] = int(
            1)  # if one value of the columns is 'yes', the value is set to yes
        df.loc[(df[list_columns[0]] == 0) & (df[list_columns[1]] == 0) &
               (df[list_columns[2]] == 0) & (df[list_columns[3]] == 0) & (
                       df[list_columns[4]] == 0), name_new_col] = int(
            0)  # else, all are 'no' so the value is set to 'no'

    elif nb_col == 6:
        df.loc[(df[list_columns[0]] == 1) | (df[list_columns[1]] == 1) | (df[list_columns[5]] == 1) |
               (df[list_columns[2]] == 1) | (df[list_columns[3]] == 1) | (
                       df[list_columns[4]] == 1), name_new_col] = int(
            1)  # if one value of the columns is 'yes', the value is set to yes
        df.loc[(df[list_columns[0]] == 0) & (df[list_columns[1]] == 0) & (df[list_columns[5]] == 0) &
               (df[list_columns[2]] == 0) & (df[list_columns[3]] == 0) & (
                       df[list_columns[4]] == 0), name_new_col] = int(
            0)  # else, all are 'no' so the value is set to 'no'
    else:
        return f"please modify the function to take into consideration a list of {nb_col} columns"
    return df


def fill_df_time_evolution(df, list_col_label, manage_lost_follow=False):
    """
    This function fills the NaN in the dataframe where the columns have been selected

    If manage_lost_follow = False, it will complete them by the last value only if they are not lost to follow up

    If manage_lost_follow = True, the patients lost to follow up are considered with the same diagnosis than before
    """

    df_new = deepcopy(df)
    bool_df = df_new.isna()
    for j, col in enumerate(list_col_label[1:]):
        col_before = list_col_label[j]
        j = j + 1

        for i in df_new.index:

            value = df_new.loc[i, col]
            # the case the patients lost to follow up stay in the label 'lost follow up'
            if not manage_lost_follow:
                if bool_df.loc[i, col]:
                    # we check the column before (ie 6 months earlier), if the patient was not lost follow up,
                    # we suppose he is in the same stage of illness
                    if df_new.loc[i, col_before] != 7:
                        df_new.loc[i, col] = df_new.loc[i, col_before]
            # the patient 'lost follow up' are considered as the same diagnosis than before
            elif manage_lost_follow:
                if df_new.loc[i, col] == 7:
                    df_new.loc[i, col] = df_new.loc[i, col_before]
                if bool_df.loc[i, col]:
                    df_new.loc[i, col] = df_new.loc[i, col_before]

    return df_new


def select_patient_class_changed(df):
    list_pat_change_from_no_risk_to_risk = []
    list_pat_change_from_no_risk_to_psychose = []
    list_pat_change_from_risk_to_psychose = []
    col_max = df.columns[-1]
    col_min = df.columns[0]

    for i in df.index:
        if df.loc[i, col_min] != df.loc[i, col_max]:  # the class has changed
            if df.loc[i, col_max] == 5 and df.loc[i, col_min] == 1:
                list_pat_change_from_no_risk_to_psychose.append(i)
            elif df.loc[i, col_max] == 5 and df.loc[i, col_min] == 3:
                list_pat_change_from_risk_to_psychose.append(i)
            elif df.loc[i, col_max] == 3 and df.loc[i, col_min] == 1:
                list_pat_change_from_no_risk_to_risk.append(i)

    return (list_pat_change_from_no_risk_to_risk,
            list_pat_change_from_no_risk_to_psychose,
            list_pat_change_from_risk_to_psychose)
