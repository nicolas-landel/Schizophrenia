import os
import pandas as pd

from Schizophrenia.application.preprocessing.utils import lower_text_column, word_in_column, creation_col_caarms, \
    select_label, fill_df_time_evolution, select_patient_class_changed, emptyVal, fill_Na_value, \
    creation_dic_matching_names, add_binary_column_to_df


def pipeline_preprocessing(name_file, option_patients_lost, period):
    """
    This function gathers all the functions for the preproceesing. It acts like a pipeline.

    It loads the data according to the name of the file in the repository 'name_file', treats it and return 2 dataframes,
    df_data and df_label.

    option_patients_lost is an int, it must be equal to 2 (keep the patients lost of follow up) or 3 (drop them)

    period (int between 0 and 4)

    """

    # load the file
    path = os.getcwd()
    df = pd.read_csv(path + '/' + name_file)

    # lower text and create new cols for psychiatrique and psychologique
    df = lower_text_column('q40', df)
    word_in_column("psychiatrique", 'q40', 'suivi psychiatrique', df)
    word_in_column("psychologique", 'q40', 'suivi psychologique', df)

    # score caarms

    df_caarms = creation_col_caarms(df, period)  # possibility of chosen the period, here t0
    df = pd.concat([df, df_caarms], axis=1)

    # selection columns
    list_columns_no_label = ['q1', 'q2', 'q8', 'q9', 'q10', 'q13', 'q15', 'q16', 'q21', 'q27', 'q29', 'q30', 'q31',
                             'q32', 'q38', 'q39', 'q42', 'q43', 'q44', 'q45', 'q46', 'q47', 'q51', 'q52', 'q53',
                             'q54', 'q55', 'q56', 'q57', 'q58', 'q59', 'q60', 'q61', 'q62', 'q63', 'q64', 'q65',
                             'q66', 'q67', 'q68', 'q69', 'q70', 'q71', 'q72', 'suivi psychiatrique',
                             'suivi psychologique', 'score caarms']

    # Columns name for the label of each consultation
    list_time_label = ['q80bis', 'q88bis', 'q97bis', 'q104bis','q112bis']
    list_all = list_columns_no_label + list_time_label
    df = df.loc[:, list_all]

    # deal with 'lost to follow up'
    if option_patients_lost == 2:
        manage_lost_ = True  # will replace the value 7 in the label by the value before (1st diagnosis)
        only_lost_patients = [
            'q80bis']  # ie just the label at T0, we will only delete patients lost at the beginning of the study
    elif option_patients_lost == 3:
        only_lost_patients = list_time_label
        manage_lost_ = True  # no matter because the patients lost have been dropped
        # ie all the label columns, we will drop patients lost at any period of the study
        list_time_class = list_time_label

    list_dropped_no_lost = []
    list_dic_lost = []

    for col in list_time_label:  # for the columns of label, depending of the option
        if col in only_lost_patients:  # cols where the patients lost will be dropped
            df, dic_lost, list_temp = select_label(df, col,
                                                   drop_lost=True)  # the patients lost to follow up are dropped
            list_dropped_no_lost = list_dropped_no_lost + list_temp
            list_dic_lost.append(dic_lost)
        else:  # cols where we will keep the patients lost
            df, dic_lost, list_temp = select_label(df, col,
                                                   drop_lost=False)  # the patients lost to follow up are dropped
            list_dropped_no_lost = list_dropped_no_lost + list_temp
            list_dic_lost.append(dic_lost)

    df = fill_df_time_evolution(df, list_time_label, manage_lost_follow=manage_lost_)  # fill the empty label

    change_class_no_lost = select_patient_class_changed(
        df[list_time_label])  # tuple of list of patients who changed of class

    df.index = [i for i in range(df.shape[0])]  # reindex because rows have been deleted with select_label

    # empty values
    count, dic = emptyVal(df)

    # this list has been done by looking into the values, it may change depending on the data

    dic_columns_completed_by_value = {'q8': 1.0, 'q9': 1.0, 'q10': 1.0, 'q13': 1.0, 'q15': 0.0, 'q16': 0.0,
                                      'q21': 0.0, 'q27': 0.0, 'q29': 1.0, 'q32': 0.0, 'q42': 0.0, 'q43': 0.0,
                                      'q44': 0.0, 'q45': 0.0, 'q46': 0.0, 'q47': 0.0, 'q56': 0.0, 'q80bis': 3,
                                      'q30': 1, 'q31': 0, 'q38': 5, 'q39': 5, 'q51': df['q51'].describe()['mean'],
                                      'score caarms': df['score caarms'].mean()}

    dic_common_value = {}
    for i, key in enumerate(dic_columns_completed_by_value.keys()):
        df, val = fill_Na_value(df, key, dic_columns_completed_by_value[key])
        dic_common_value[key] = val

    # inversion yes_no where yes=0 and no=1 in the data to match with the standard yes=1 and no=0
    list_column_inversion_yes_no = ['q13']
    for col in list_column_inversion_yes_no:
        df[col] = (df.loc[:, col] - 1) ** 2

    # rename the columns and the label

    column_names = ["sexe", "age", "vit avec", "vit en milieu urbain", "niveau scolaire", "redoublement", "emploi",
                    "addresse par",
                    "anomalies durant la grossesse", "consommation toxiques durant grossesse",
                    "interaction mere-enfant correct",
                    "lien au pere correct", "developpement psychoaffectif", "antecedents judiciaires",
                    "traitements psychotropes actuels", "traitements psychotropes passes",
                    "maltraitance physique familiale", "maltraitance verbale familiale", "carence affective",
                    "carence educative",
                    "abus sexuel", "deces d un parent", "score sofas", "EDM actuel ou passe au MINI",
                    "EDM avec melancholique actuel",
                    "dysthymie actuelle", "risque sucidaire actuel",
                    "episode hypomaniaque", "episode maniaque", "trouble panique sans agoraphobie",
                    "agoraphobie sans antecedent de trouble de panique",
                    "phobie sociale", "TOC", "PTSD",
                    "dependance alcool", "dependance opiace", "dependance cocaine", "dependance cannabis",
                    "dependance sedatifs",
                    "dependance hallucinogenes", "anorexy mentale", "boulemie", "anxiete generalisee",
                    "personnalite antisociale",
                    "suivi psychiatrique", "suivi psychologique", "score caarms"]

    label_names = ['label', 'label_6mois', 'label_12mois', 'label_18mois', 'label_24mois']

    all_names = column_names + label_names  # list of the names of the columns, X (data) and Y (label)

    # to store the information of the matches between names and columns
    dic_matching_names = creation_dic_matching_names(df, all_names)
    df.columns = all_names  # rename columns

    # creation new columns : combinaition of others (dependance and traumatisme)
    list_dependance = ['dependance alcool', 'dependance opiace', 'dependance cocaine', 'dependance cannabis',
                       'dependance sedatifs', 'dependance hallucinogenes']
    list_trauma = ['maltraitance physique familiale', 'maltraitance verbale familiale', 'carence affective',
                   'carence educative', 'abus sexuel']

    # creation new feature 'dependance' based on features in list_dependance
    df = add_binary_column_to_df(df, list_dependance, 'dependance')
    df = add_binary_column_to_df(df, list_trauma, 'traumatisme')  # same for traumatisme

    # split data and label
    df_label = df.loc[:, label_names]
    df_data = df.drop(label_names, axis=1)

    # return the df with the data and the df with the label
    return df_data, df_label
