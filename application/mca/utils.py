from copy import deepcopy

import pandas as pd
import mca
import numpy as np


def select_max_positive_contribution_fs(table, factor, nb_values, name_value):
    """
    This function selects the nb_values (int) max values of the table for the row fs (int) and return the data
    with the modalities as first columns and the values as second columns. You have to name the 2nd column (name_value, string)

    """
    # We enter the facteur as the id of the factor resulting of the mca but here we need to deal with the position in
    # the index of the table, starting at 0. So the factor "1" is represented in the row 0
    try:
        factor = factor - 1
    except:
        # todo bare except
        print('erreur avec le facteur')
    if table.columns[-1] == ('label', 'psychose'):
        max_list = sorted(zip(table.iloc[factor][:-3], table.columns[:-3]), reverse=True)[:nb_values]
        values = [max_list[i][0] for i in range(len(max_list))]
        columns = [max_list[i][1] for i in range(len(max_list))]
    else:
        max_list = sorted(zip(table.iloc[factor], table.columns), reverse=True)[:nb_values]
        values = [max_list[i][0] for i in range(len(max_list))]
        columns = [max_list[i][1] for i in range(len(max_list))]
    df_max = pd.DataFrame(columns=['modalités', name_value])
    df_max['modalités'] = [str(columns[i][0]) + ': ' + str(columns[i][1]) for i in range(len(columns))]
    df_max[name_value] = values
    return df_max


def select_max_negative_contribution_fs(table, factor, nb_values, name_value):
    try:
        factor = factor - 1
    except:
        # todo bare except
        print('erreur avec le facteur')
    if table.columns[-1] == ('label', 'psychose'):
        max_list = sorted(zip(table.iloc[factor][:-3], table.columns[:-3]), reverse=False)[:nb_values]
        values = [max_list[i][0] for i in range(len(max_list))]
        columns = [max_list[i][1] for i in range(len(max_list))]
    else:
        max_list = sorted(zip(table.iloc[factor], table.columns), reverse=False)[:nb_values]
        values = [max_list[i][0] for i in range(len(max_list))]
        columns = [max_list[i][1] for i in range(len(max_list))]
    df_min = pd.DataFrame(columns=['modalités', name_value])
    df_min['modalités'] = [str(columns[i][0]) + ': ' + str(columns[i][1]) for i in range(len(columns))]
    df_min[name_value] = values
    return df_min


def translate_contribution_to_sentence(df_contribution, table_modalities_mca, factor):
    sentence_positive = 'Sur la droite de l\'axe du facteur ' + str(
        factor) + ' seront principalement regroupés les patients ayant comme valeurs : '
    sentence_negative = 'Sur la gauche du l\'axe du facteur ' + str(
        factor) + ' seront principalement regroupés les patients ayant comme valeurs : '

    for i, modality in enumerate(df_contribution['modalités']):
        modality_table_mca = (str(modality.split(':')[0]), str(modality.split(':')[1][1:]))

        if table_modalities_mca.loc[('Factor', factor), modality_table_mca] >= 0:
            sentence_positive = sentence_positive + str(modality) + '; '
        elif table_modalities_mca.loc[('Factor', factor), modality_table_mca] <= 0:
            sentence_negative = sentence_negative + str(modality) + '; '

    return sentence_positive, sentence_negative


def apply_mca (df, benzecri):
    """
    This function creates the object MCA : it applies a multiple analysis components on a disjunctive array
    A MCA will try to create new features by combining the former ones in order to have the fewer new features keeping the maximum information (test chi2)
    The correction of benzecri is
    """
    # number of variables in data_mca (here 46), maybe there is a faster way to calculate it
    ncols = len(df.columns.get_level_values(0).unique())
    mca_ = mca.MCA(df, ncols=ncols, benzecri=benzecri)
    return mca_


def apply_mca_df_patient(df, nb_factors=10):
    """
    This function applies the mca to the dataframe df (usually data_mca). The benzecri correction is not applied.
    It returns the dataframe of the patients as column and factor as index and where the values are the projections
    of the patients onto this factor. If the value of projection is high, this means the values of the patient is the
    different modalities is well represented by the factor (which is a combination of some modalities).
    """
    mca_ben = apply_mca(df, False)  # benzecri correction is not applied

    fs = 'Factor'  # fs are the factor
    table_patients_mca = pd.DataFrame(columns=df.index,
                                      index=pd.MultiIndex.from_product([[fs], range(1, nb_factors + 1)]))

    table_patients_mca.loc[fs, :] = mca_ben.fs_r(N=nb_factors).T  # add the N=10 first factor to the table_patients_mca

    table_patients_mca = - np.round(table_patients_mca.astype(float), 2)

    return table_patients_mca


def apply_mca_df_patient_time(list_df_, index_period, nb_factors=10, benzecri=False):
    """ This function takes a list of df (disjunctive arrays), the index period (int between 0 and 4) and the nb of factors.

    It will apply the mca without the benzecri coeff

    It returns
    """
    list_df = deepcopy(list_df_)  # because the list is poped so it avoids an empty list
    df = list_df.pop(index_period)
    # number of variables in data_mca (here 46), maybe there is a faster way to calculate it
    ncols = len(df.columns.get_level_values(0).unique())
    mca_ben = mca.MCA(df, ncols=ncols, benzecri=benzecri)  # benzecri correction can be applied
    fs = 'Factor'  # fs are the factor
    table_patients_mca = pd.DataFrame(columns=df.index,
                                      index=pd.MultiIndex.from_product([[fs], range(1, nb_factors + 1)]))
    nb_patients = df.shape[0]
    table_patients_mca.loc[fs, :] = mca_ben.fs_r(N=nb_factors).T  # add the N=10 first factor to the table_patients_mca
    # table_patients_mca = table_patients_mca #because their is an inversion of sign

    if index_period == 0:
        for t, df_ in enumerate(list_df):
            for i in df_.index:
                temp_array = np.array(df_.iloc[i])
                # print(mca_ben.fs_r_sup(pd.DataFrame([temp_array]), N=nb_factors))

                table_patients_mca.loc[fs, str(i) + '_t' + str(t + 1)] = - mca_ben.fs_r_sup(pd.DataFrame([temp_array]), N=nb_factors)[0]

    if index_period != 0:
        # rename the columns
        table_patients_mca.columns = [str(table_patients_mca.columns[i]) + '_t' + str(index_period) for i in
                                      range(table_patients_mca.shape[1])]  # rename the columns
        for t, df_ in enumerate(list_df):
            for i in df_.index:
                if t == 0:  # if it s the 1st period, the name of the patient has no suffix
                    table_patients_mca.loc[fs, str(i)] = - mca_ben.fs_r_sup(df_, N=nb_factors)[i]
                elif t != 0:
                    table_patients_mca.loc[fs, str(i) + '_t' + str(t)] = - mca_ben.fs_r_sup(df_, N=nb_factors)[i]

        # not the good order of time
        cols = table_patients_mca.columns

        columns_patients_period = cols[: nb_patients]
        columns_other_periods = cols[nb_patients:]
        new_cols = columns_patients_period.to_list() + columns_other_periods.to_list()
        table_patients_mca = table_patients_mca[new_cols]

    table_patients_mca = np.round(table_patients_mca.astype(float), 2)

    # split into 5 df for each period
    list_tables_patients = []
    for k in range(5):
        temp_df = table_patients_mca.iloc[:, k * nb_patients:(k + 1) * nb_patients]
        list_tables_patients.append(-temp_df)

    return list_tables_patients


def apply_mca_df_modalities(df, df_label_disj_, nb_factors):
    """
    This function calculates the projection of the modalities onto the factors of the mca. If the value of the
    projection is high, this means the modality has a high contribution to this axis.

    """

    # table_modalities_mca is the dataframe where the categories are projected onto the factor
    # the benzecri coeff is used

    ncols = len(df.columns.get_level_values(
        0).unique())  # number of variables in data_mca (here 46), maybe there is a faster way to calculate it
    # benzecri correction is applied, the eigenvalues below 1/K are dropped (where K is the number of variables,
    # here 46 (sexe, age, ...)). A coefficient with a factor K/(K-1) is also applied to the remaining variables.
    mca_ben = mca.MCA(df, ncols=ncols, benzecri=True)

    fs = 'Factor'

    table_modalities_mca = pd.DataFrame(columns=df.columns,
                                        index=pd.MultiIndex.from_product([[fs], range(1, nb_factors + 1)]))
    # print(table_modalities_mca.shape, mca_ben.fs_c(N=20).T.shape, mca_ben.fs_c(N=20).T)
    table_modalities_mca.loc[fs, :] = mca_ben.fs_c(N=nb_factors).T  # selection of the N=10 first factor
    # projection of the modalities of the label ('a risque', etc) onto the factors
    fs_c_sup = mca_ben.fs_c_sup(df_label_disj_, N=nb_factors)

    table_modalities_mca.loc[fs, ('label', 'pas de risque')] = fs_c_sup[0]
    table_modalities_mca.loc[fs, ('label', 'a risque')] = fs_c_sup[1]
    table_modalities_mca.loc[fs, ('label', 'psychose')] = fs_c_sup[2]

    table_modalities_mca = np.round(table_modalities_mca.astype(float), 2)

    return table_modalities_mca


def apply_mca_df_modalities_contributions(df, nb_factors):
    """ This function calculates the contributions of the modalities for the factors of the mca. If the value of the
    projection is high, this means the modality has a high contribution to this axis.

    """

    # table_modalities_mca is the dataframe where the categories are projected onto the factor
    # the benzecri coeff is used

    ncols = len(df.columns.get_level_values(
        0).unique())  # number of variables in data_mca (here 46), maybe there is a faster way to calculate it
    # benzecri correction is applied, the eigenvalues below 1/K are dropped
    # (where K is the number of variables, here 46 (sexe, age, ...)).
    # A coefficient with a factor K/(K-1) is also applied to the remaining variables.
    mca_ben = mca.MCA(df, ncols=ncols,benzecri=True)

    table_modalities_mca_contribution = pd.DataFrame(columns=df.columns, index=pd.MultiIndex.from_product(
        [['contributions'], range(1, nb_factors + 1)]))
    # print(table_modalities_mca.shape, mca_ben.fs_c(N=20).T.shape, mca_ben.fs_c(N=20).T)
    table_modalities_mca_contribution.loc['contributions', :] = mca_ben.cont_c(N=10).T * 1000

    table_modalities_mca_contribution = np.round(table_modalities_mca_contribution.astype(float), 1)

    return table_modalities_mca_contribution


def evaluation_mca(df, nb_factors):
    """
    Eigenvalues are denoted by λ and the proportions of explained variance by τ
    the Benzecri factor is [(nb_cols/(nb_cols-1))*(λ-(1/nb_cols))]**2 , for example, if λ=0.11, the coeff is 0.008692
    the greenacre coefficient is ((nb_cols/(nb_cols-1))*(sum(λ**2)-(nb_categories-nb_cols)/(nb_cols**2))
    where nb_categories is 134. These values can be found in the following array
    """
    mca_ind = apply_mca(df, False)
    mca_ben = apply_mca(df, True)

    data = {'Iλ': pd.Series(mca_ind.L[:nb_factors]),
            # eigenvalues (valeurs propres) for the N factors witout Benzecri or Greenacre correction
            'τI': mca_ind.expl_var(greenacre=False, N=nb_factors),
            # explained variance for the factors (eigenvalue/sum(eigenvalues))
            'Zλ': pd.Series(mca_ben.L[:nb_factors]),
            # eigenvalues (valeurs propres) for the N factors with Benzecri correction and without greenacre
            'τZ': mca_ben.expl_var(greenacre=False, N=nb_factors),
            # explained variance (coeff_benzecri(eigenvalue)/sum(coeff_benzecri(eigenvalue)))
            'cλ': pd.Series(mca_ben.L[:nb_factors]),  # same data than Zλ
            # the explained variance for the factors is corrected with greenacre,
            # instead of dividing the coeff_benzecri(eigenvalue) by sum(coeff_benzecri(eigenvalue))
            'τc': mca_ind.expl_var(greenacre=True, N=nb_factors)}
    # the eigenvalue is divided by sum(coeff_greenacre(eigenvalue))

    # L : vector of the principal inertias (eigenvalues) of the factors
    # Method expl_var returns proportion of explained inertia for each factor
    # N limits number of retained factors

    # 'Indicator Matrix', 'Benzecri Correction', 'Greenacre Correction'
    columns = ['Iλ', 'τI', 'Zλ', 'τZ', 'cλ', 'τc']

    table_explained_mca = pd.DataFrame(data=np.zeros(1))  # create empty dataframe

    for key in data.keys():
        df_temp = pd.DataFrame({str(key): data[key]})  # create temporary dataframe with one key of the dictionary data
        table_explained_mca = pd.concat([table_explained_mca, df_temp], axis=1).fillna(
            0)  # and add it to the final dataframe

    table_explained_mca = table_explained_mca.drop([0], axis=1)  # the first column is just 0, it was to initate
    # table2 = pd.DataFrame(data=data, columns=columns).fillna(0)  #doesn t work
    table_explained_mca.index += 1
    table_explained_mca.loc['Σ'] = table_explained_mca.sum()  # create the sum at the last row of the df
    table_explained_mca.index.name = 'Factor'

    return table_explained_mca
