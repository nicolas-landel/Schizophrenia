from .graphs import (
    position_vector,
    creation_dataframe_distance_modalities,
    select_dist_modalities,
)


def apply_mca_analysis(fs_id1, fs_id2, dist_max, modality, table_modalities_mca, table_explained_mca, df_data_disj,
                       df_label_disj):
    """
    Pipeline of the functions above which allows to create the dataframe of the closest modalities of the modality chosen

    """
    # storing the position of the modalities in the plan of the factors fs_id1 and fs_id2
    vect, pos_x, pos_y, new_norm = position_vector(table_modalities_mca, fs_id1, fs_id2)

    # calculating the distances between the significant modalities and storing them in a upper dataframe
    df_distances = creation_dataframe_distance_modalities(pos_x, pos_y, fs_id1, fs_id2, table_modalities_mca,
                                                          table_explained_mca)

    # create a dataframe (modality, dist, effective) of the modalities having
    # a lower distance than dist_max with the modality
    df_close_modalities = select_dist_modalities(modality, df_distances, dist_max, df_data_disj, df_label_disj)

    return df_close_modalities

