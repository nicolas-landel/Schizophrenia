from .utils import (
    apply_mca, apply_mca_df_patient,
    apply_mca_df_modalities,
    apply_mca_df_modalities_contributions,
    evaluation_mca,
    apply_mca_df_patient_time,
)


def pipeline_mca(list_df_data_disj, index_period, df_label_disj, nb_factors, benzecri):
    # apply the mca and creates the tables : modalities coordo & contributions,
    # patients coordo, mca explanation for a defined period
    df = list_df_data_disj[index_period]
    mca = apply_mca(df, benzecri)
    df_label_disj_period = df_label_disj.iloc[:, 3 * index_period:3 * index_period + 3]

    table_patients_mca = apply_mca_df_patient(df, nb_factors)
    table_modalities_mca = apply_mca_df_modalities(df, df_label_disj_period, nb_factors)
    table_modalities_mca_contribution = apply_mca_df_modalities_contributions(df, nb_factors)
    table_explained_mca = evaluation_mca(df, nb_factors)

    # for the patients, we need to have the mca for each period
    list_table_patients_mca_time = apply_mca_df_patient_time(list_df_data_disj, index_period, nb_factors, benzecri)

    return (table_modalities_mca, table_modalities_mca_contribution, table_patients_mca,
            table_explained_mca, list_table_patients_mca_time)


