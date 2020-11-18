import plotly.graph_objects as go

def prepare_square_plotly(fs_id1, fs_id2, table):
    """ This function plots a square. It allows us to determine the modalities where the contribution ((weight*(aj)**2))/eigenvalue
    is superior to the weight(nb elements in the modality/ (nb of modalities * nb of features))

    ie : val_abs(ai)<sqrt(eigenvalue i) ie the position of the modality is superior to the square of the eigenvalue of the axe (far enough of the origin)

    """

    # todo refactor
    top_line = go.Scatter(x=[-table.loc[fs_id1, 'Zλ'] ** 0.5, table.loc[fs_id1, 'Zλ'] ** 0.5],
                          y=[table.loc[fs_id2, 'Zλ'] ** 0.5, table.loc[fs_id2, 'Zλ'] ** 0.5],
                          mode='lines',
                          line=go.scatter.Line(color="orange"))

    bottom_line = go.Scatter(x=[-table.loc[fs_id1, 'Zλ'] ** 0.5, table.loc[fs_id1, 'Zλ'] ** 0.5],
                             y=[-table.loc[fs_id2, 'Zλ'] ** 0.5, -table.loc[fs_id2, 'Zλ'] ** 0.5],
                             mode='lines',
                             line=go.scatter.Line(color="orange"))

    right_line = go.Scatter(y=[-table.loc[fs_id2, 'Zλ'] ** 0.5, table.loc[fs_id2, 'Zλ'] ** 0.5],
                            x=[table.loc[fs_id1, 'Zλ'] ** 0.5, table.loc[fs_id1, 'Zλ'] ** 0.5],
                            mode='lines',
                            line=go.scatter.Line(color="orange"))

    left_line = go.Scatter(y=[-table.loc[fs_id2, 'Zλ'] ** 0.5, table.loc[fs_id2, 'Zλ'] ** 0.5],
                           x=[-table.loc[fs_id1, 'Zλ'] ** 0.5, -table.loc[fs_id1, 'Zλ'] ** 0.5],
                           mode='lines',
                           line=go.scatter.Line(color="orange"))

    return ([top_line, bottom_line, right_line, left_line])
