from sklearn.model_selection import StratifiedShuffleSplit


def split_train_test(df_data_disj, df_label, period, test_size, keep_psychose=True):
    """ test size (float) between 0.1 and 0.9"""

    if keep_psychose:
        df_data_disj_opposite = (df_data_disj - 1) ** 2  # 0 where is true, 1 where is false

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

        for train_index, test_index in split.split(df_data_disj_opposite, df_label.iloc[:, period]):
            x_train = df_data_disj_opposite.loc[train_index]
            y_train = df_label.iloc[train_index, period]
            x_test = df_data_disj_opposite.loc[test_index]
            y_test = df_label.iloc[test_index, period]

    elif keep_psychose:
        df_data_disj_opposite = (df_data_disj - 1) ** 2
        list_index_psychose = [df_label.index[i] for i in range(df_label.shape[0]) if df_label.iloc[i, period] == 5]

        df_label_ = df_label.drop(list_index_psychose, axis=0)

        df_data_disj_opposite = df_data_disj_opposite.drop(list_index_psychose, axis=0)

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        df_label_.index = [i for i in range(df_label_.shape[0])]
        df_data_disj_opposite.index = [i for i in range(df_data_disj_opposite.shape[0])]

        for train_index, test_index in split.split(df_data_disj_opposite, df_label_.iloc[:, period]):
            x_train = df_data_disj_opposite.loc[train_index]
            y_train = df_label_.iloc[train_index, period]
            x_test = df_data_disj_opposite.loc[test_index]
            y_test = df_label_.iloc[test_index, period]

    return x_train, y_train, x_test, y_test
