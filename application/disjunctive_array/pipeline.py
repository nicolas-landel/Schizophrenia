from copy import deepcopy

import pandas as pd

from .utils import creation_disj_df_threshold_split, \
    creation_disjunctive_array, creation_dic_nb_val_by_col, split_columns, creation_list_variables_repeted, \
    creation_dic_nb_val_by_col_label


def pipeline_disjunctive_df_data(df_data):
    """
    This function creates the disjunctive dataframe df_data_disj
    """

    # split numerical features into 2 according the values are above or below a threshold
    df_age = creation_disj_df_threshold_split(df_data, 'age', 20, 'age superieur a', 'age inferieur a')
    df_sofas = creation_disj_df_threshold_split(df_data, 'score sofas', 70,
                                                'score sofas superieur a', 'score sofas inferieur a')

    # for score caarms (numerical), we do a multiple split into 4 bins instead a threshold
    list_name_, list_var_ = ['score caarms'], [df_data['score caarms']]
    # you can change the number of bins for score caarms here
    disj_df_caarms, col_var_num = creation_disjunctive_array(4, list_name_, list_var_)

    # df_preparation_mca is the df df_data where the features are categorial and will be split
    # we are going to modify the dataframe but we also need to keep a version of it intact so deepcopy
    df_preparation_mca = deepcopy(df_data)
    df_preparation_mca = df_preparation_mca.drop(['age', 'score sofas', 'score caarms'], axis=1)

    # creation of the dic which stores the number of modalities by feature
    # the dic_nb_val is created according the automatic function where the number
    dic_nb_val = creation_dic_nb_val_by_col(df_preparation_mca)

    # the columns of df_preparation_mca are split according to their modalities
    df_split = split_columns(df_preparation_mca, dic_nb_val)

    # creation of the multiindex for the data
    var_index = creation_list_variables_repeted(dic_nb_val)

    # definition of the modalities for each feature as list of string format
    gender = ['M', 'F']
    yn = ['oui', 'non']
    ny = ['non', 'oui']
    study_level = ['primaire', 'secondaire', 'superieur', 'BEP', 'CAP', 'autre']
    live_with = ['pere', 'mere', 'enfants', 'fratrie', 'conjoint', 'seul', 'autre famille', 'ami', 'sans domicile',
                 'coloc', 'institution', 'famille d accueil', 'autre']
    job = ['etudiant', 'CDD', 'CDI', 'invalidité', 'Aah', 'RSA', 'autre']
    referred_by = ['med psychiatre', 'psychologue', 'med traitant', 'educateur', 'famille', 'med scolaire', 'autre']
    dvpt_psycho = ["normal", "perturbe"]
    psycho_treatment = ['antidepresseur', 'antipsychotiques', 'anxiolytiques', 'thymoregulateur', 'hypnotiques',
                        'autre', 'aucun']
    sucidal_risk = ["leger", "moyen", "haut"]

    # creation of a list of all the modalities for all the features,respecting the order to match with the features
    val_index = gender + live_with + ny + study_level + ny + job + referred_by + ny * 4 + dvpt_psycho \
                + ny + psycho_treatment * 2 + ny * 6 + \
                ny * 3 + sucidal_risk + ny * 17 + ny * 4

    # creation of the multiindex
    col_index = pd.MultiIndex.from_arrays([var_index, val_index], names=['features', 'modalities'])

    # concatenation and rename the columns with the multiindex
    data_mca = pd.DataFrame(data=df_split.values, index=df_split.index,
                            columns=col_index)  # df_mca is the disjunctive array
    data_mca = pd.concat([data_mca, df_age, df_sofas, disj_df_caarms], axis=1)
    return data_mca


def pipeline_disjunctive_df_label(df_label):
    """ This function creates the disjunctive dataframe df_label_disj"""

    # creation of a dic where the keys are the columns of the df label and the values are 3 (because 3 modalities/class)
    dic_nb_val = creation_dic_nb_val_by_col_label(df_label)
    # split the df into a new df
    df_label_split = split_columns(df_label, dic_nb_val, label=True)

    # creation multiindex
    var_index = creation_list_variables_repeted(dic_nb_val)
    class_label = ['pas de risque', 'a risque', 'psychose']
    val_index = class_label * df_label.shape[1]
    col_index = pd.MultiIndex.from_arrays([var_index, val_index], names=['features', 'modalities'])

    df_label_disj = pd.DataFrame(data=df_label_split.values, index=df_label_split.index, columns=col_index)

    return df_label_disj
