from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from Schizophrenia.application.disjunctive_array.utils import creation_dic_nb_val_by_col_label, split_columns


def chi2_table(df):
    """ This function creates 2 data frames, one with the chi2 coefficients of the chi2 test on all the variables, the other
    is the p_values of each chi2 test. The format of each data frame is all the columns' names x all the columns' names (50x50 here)

    It takes in argument the data frame df_test_chi2 which is the dataframe df_mca plus the label in a disjunctive format

    It creates cross tables for each pairs of variables and applies the chi2 test from the module scipy
    """

    df_chi2 = pd.DataFrame()
    df_p_chi2 = pd.DataFrame()
    for j, var1 in enumerate(df.columns):
        df_chi2[var1] = [i * 0 for i in range(df.shape[1])]
        df_p_chi2[var1] = [i * 0 for i in range(df.shape[1])]

        for i, var2 in enumerate(df.columns):
            table = pd.crosstab(df[var1].to_numpy(), df[
                var2].to_numpy())  # I had to add .to_numpy() otherwise it doesn't work with last verion of pandas
            # whereas it worked before (strange ..)
            chi2, p, dof, expected = chi2_contingency(table.values)
            df_chi2.loc[i, var1] = round(chi2, 3)
            df_p_chi2.loc[i, var1] = round(p, 3)
    df_chi2.index = df.columns
    df_p_chi2.index = df.columns
    return (df_chi2, df_p_chi2)


def p_value_inf(p_val_lim, df):
    """ This function will selected the p_values in the dataframe df which are lower than the p_val_lim you set
    It will return the dataframe with value=True if the p_value for the 2 features is lower, else value=None.

    Not really useful function ..
    """

    new_df = deepcopy(df)

    for col in df.columns:
        new_df.loc[new_df[col] > p_val_lim, col + ' correlated'] = None
        new_df.loc[new_df[col] <= p_val_lim, col + ' correlated'] = True

        new_df = new_df.drop([col], axis=1)

    return (new_df)


def correlation_revealed(df_test_chi2, p_val_lim, min_value_each_col, df, khi2_label=False):
    """ This function selects the columns where the p_values is smaller than p_val_lim (usually 0.05) so the hypothesis
    of correlation is validated

    It returns la list of tuple (the 2 variables correlated)

    I have added the option which allows to not select the columns where there are fewer than 'min_value_each_col' for
    the binary answers ('0/1'). It can be set to 5 for example

    """

    list_corre = []
    for col in df.columns:
        bins_var, delimitation_bins = np.histogram(df_test_chi2[col], 2)
        for ind in df.index:
            bins_var_2, delimitation_bins_2 = np.histogram(df_test_chi2[ind], 2)
            # conditions to be correlated
            if df.loc[ind, col] <= p_val_lim and col != ind and (ind, col) not in list_corre:
                # condition added to limit columns with too few 'yes' or few 'no'
                if bins_var[0] >= min_value_each_col and bins_var[1] >= min_value_each_col and\
                        bins_var_2[0] >= min_value_each_col and bins_var_2[1] >= min_value_each_col:

                    if ('pas de risque' in ind and 'a risque' in col) or ('pas de risque' in col and 'a risque' in ind):
                        pass
                    elif ('psychose' in ind and 'a risque' in col) or ('psychose' in col and 'a risque' in ind):
                        pass
                    elif ('psychose' in col and 'pas de risque' in ind) or (
                            'pas de risque' in col and 'psychose' in ind):
                        pass
                    elif ('psychose' in col and 'psychose' in ind) or\
                            ('pas de risque' in col and 'pas de risque' in ind):
                        pass
                    elif 'a risque' in col and 'a risque' in ind:
                        pass
                    elif khi2_label and (
                            ('pas de risque' not in col and 'a risque' not in col and 'psychose' not in col)
                            and ('pas de risque' not in ind and 'a risque' not in ind and 'psychose' not in ind)):
                        pass
                    else:
                        list_corre.append((col, ind))
    return list_corre


def modify_df_label_chi2(df_label):
    """

    :param df_label:
    :return:
    """
    dic_label = creation_dic_nb_val_by_col_label(df_label)  # work to do here
    df_label_split = split_columns(df_label, dic_label, label=True)
    df_label_split.columns = ['pas de risque t0', 'a risque t0', 'psychose t0',
                              'pas de risque t1', 'a risque t1', 'psychose t1',
                              'pas de risque t2', 'a risque t2', 'psychose t2',
                              'pas de risque t3', 'a risque t3', 'psychose t3',
                              'pas de risque t4', 'a risque t4', 'psychose t4']

    return df_label_split
