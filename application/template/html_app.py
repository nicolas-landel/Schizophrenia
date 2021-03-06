import math
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import ast
import base64
from copy import deepcopy
from dash.dependencies import Input, Output, State



from disjunctive_array.pipeline import (pipeline_disjunctive_df_data,
                                        pipeline_disjunctive_df_label)
from ml.utils import split_train_test
from preprocessing import pipeline_preprocessing
from process_mca.pipeline import pipeline_mca
from process_mca.utils import (select_max_positive_contribution_fs,
                               select_max_negative_contribution_fs,
                               translate_contribution_to_sentence)
from visualisation.graphs import (creation_dataframe_distance_modalities,
                                  interactive_plot_variable_by_variable,
                                  apply_color_label,
                                  position_vector, select_dist_modalities,
                                  interactive_plot_patient_modality,
                                  interactive_plot_patient_time,
                                  interactive_plot_patient_time_follow_3d)
from ml.decision_tree import best_param_tree, plot_tree, delete_col_decision_tree
from ml.random_forest import random_forest_grid_search, creation_random_forest, evaluate_random_forest
from statistics.chi2 import (chi2_table, correlation_revealed,
                             modify_df_label_chi2)

class GenerateApp():
    """
    """
    classes_name_with_psy = ["pas de risque","a risque","psychose"]
    classes_name_without_psy = ["pas de risque","a risque"]
    list_moda = [('label', 'psychose')]  #default list_modalities
    period = 4
    test_size = 0.2
    df_distances = pd.DataFrame()
    counter = 0

    def __init__(self, *args, **kwargs):
        print("INIT KWARGS", kwargs)
        self.app = kwargs.get('app')

        self.process_pipelines()

        self.html = [
            html.H1(className="app-title", children='Prévention en santé mentale'),
            html.Div(id="all-the-data",
                className="all-the-data",
                children=[ 
                    html.Div(id='param-global-var',
                        children=[
                            html.H2(
                                children="Paramètres de l'application"
                            ),
                            html.P(children="Veuillez choisir le nom du fichier csv dans le répertoir courant :", className="question"),
                            dcc.Input(
                                id='name_file',
                                value='raw_data.csv',
                                type='text'
                            ),
                            html.P(className="question",
                                children="Voulez-vous conserver les patients perdus de vue au cours de l'étude \
                                (après avoir au moins effectués la première consultation) en les \
                                considérant avec le même diagnostic qu'à leur dernière consultation ou les enlever de l'étude ?",
                            ),
                            dcc.RadioItems(
                                id='option_lost',
                                options=[
                                    {'label':'garder les patients perdu de vue', 'value':2},
                                    {'label':'retirer les patients perdus de vue ', 'value':3}],
                                value=2,
                                labelStyle={'display': 'inline-block'},
                                className="answers"
                            ),
                            html.P(
                                className="hidden-div",
                                id='hidden_div'
                            ),
                        ]
                    ),
                    html.Div(id="test chi2",
                        className="new-section",
                        children=[
                            html.H2(children="Test du chi2"),
                            html.P(
                                className="explanation",
                                children="Le test du chi2 permet de savoir si deux variables qualitatives sont indépendantes ou non, c'est à dire si les réponses de l'une dépendent de l'autre.\
                                        On formule l'hypothèse H0 : il y a indépendance entre les 2 variables. On fixe un seuil de confiance, habituellement 0.05.\
                                        Les variables qui sont affichées ci-dessous ont toutes une hypothèse H0 rejetée. Elles ne sont donc pas indépendantes.",
                            ),
                            html.P(className="question",
                                children="Veuillez choisir un seuil de confiance et le nombre minimal d'éléments \
                                qu'il doit y avoir dans la plus petite des modalités (pour les variables catégorielles de type 'oui/non') :"
                            ),
                            dcc.Input(
                                id="threshold_chi2",
                                value="0.05",
                                type="text"
                            ),
                            dcc.Input(
                                id="minimum_elements", 
                                value="5", 
                                type="text"
                            ),
                            html.P(className="question",
                                children="Voulez-vous faire le test du khi2 uniquement entre les variables et le label ?"),
                            dcc.RadioItems(
                                id="chi2_label",
                                options=[
                                    {'label': 'oui', 'value': 1},
                                    {'label': 'non', 'value': 0}
                                    ],
                                value=0,
                                labelStyle={'display': 'inline-block'}
                            ),
                            html.P(
                                id="explication-chi2", 
                                className="explanation", 
                                children="Voici ci dessous la liste des variables qui sont \
                                    associées entre elles (2 par 2). On ne peut cependant pas savoir dans quel sens varie cette dépendance."
                            ),
                            html.Div(
                                id='resultat chi2', 
                                className="results-chi2",
                                children=[
                                    html.P(id="list of correlation1"),
                                    html.P(id="list of correlation2"),
                                    html.P(id="list of correlation3")
                                ]
                            )           
                        ],
                    ),
                    html.Div(id="all the ACM",
                        className="new-section",
                        children=[
                            html.H2(children="Analyse des correspondances multiples"),
                            html.P(children="Effectif d'une modalité", className="question"),
                            dcc.Dropdown(
                                id="effective_modality",
                                options=[{'label':str(i) ,'value': str(i)} for i in self.df_data_disj.columns],
                                placeholder="Choisissez une modalité"     
                            ),
                            html.P(id='effective_modality_display'),

                            html.Div(id="1er graphe ACM",
                                children=[
                                    html.H3(children="Graphe des coordonnées des modalités sur le plan de facteurs de l'ACM "),

                                    html.Div(id = 'graph_var_by_var',
                                        children=[
                                            html.P(className="question",
                                                children='Quel référentiel de temps voulez vous choisir pour le label ?'
                                            ),
                                            dcc.RadioItems(
                                                id='choose_period_graph',
                                                options=[
                                                    {'label':'M0', 'value':0},
                                                    {'label':'M6', 'value':1},
                                                    {'label':'M12', 'value':2},
                                                    {'label':'M18', 'value':3},
                                                    {'label':'M24', 'value':4}
                                                ],
                                                value=0,
                                                labelStyle={'display' : 'inline-block'}
                                            ),
                                            html.P(children="Choisissez le facteur pour l'axe des abscisses", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id1_',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=1,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Choisissez le facteur pour l'axe des ordonnées", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id2_',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=2,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Voulez-vous afficher uniquement les modalités significatives ?", className="question"),
                                            dcc.RadioItems(
                                                id='significant_only',
                                                options=[
                                                    {'label': "oui", 'value': 1},
                                                    {'label':"non", 'value': 0}
                                                ],
                                                value=0,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.Button(
                                                id='submit-button-state_3', n_clicks=0, children='Afficher le graphe'
                                            ),
                                            dcc.Loading(
                                                id="loading-3",
                                                type="default",
                                                children= dcc.Graph(
                                                    id='var_by_var', 
                                                    config ={'autosizable' :True, 'responsive':False},
                                                    style={"postion":"relative"}
                                                )
                                            )                                             
                                        ]
                                    )
                                ]
                            ),
                            html.Div(id="explanation graphe modalities",
                                children=[
                                    html.H4(children="Analyse des coordonées des modalités du graphe"),
                                    html.Div(id="hidden_div4", style={"display":"none"}),
                                    html.P(children="Veuillez choisir une modalité dont on cherchera les modalités proches dans le graphe", className="question"),

                                    dcc.Dropdown(
                                        id="modality_to_evaluate",
                                        value=('label','psychose') #value by default
                                    ),
                                    html.P(className="question", children="Veuillez choisir la distance euclidienne maximum avec les autres modalités"),

                                    dcc.Slider(
                                        id='dist_moda',
                                        min=0.0,
                                        max=20,
                                        marks={i: '{}'.format(round(i/10,1)) for i in range(0,21)},
                                        value= 6
                                    ),
                                    
                                    html.H4(id="text_before_array"),
                                    dcc.Loading(
                                        id="loading-4",
                                        type="default",
                                        className="margin-30",
                                        children= html.Div(
                                            className="array",
                                            id='array_graph_moda'
                                        )
                                    )
                                ]
                            ),
                            html.Div(id='analysis factors',
                                children=[
                                    html.H4(children="Analyse d'un facteur"),
                                    html.P(children='Choisissez le facteur que vous vouler étudier', className="question"),
                                    dcc.RadioItems(
                                        id='factor_to_analyse',
                                        options=[{'label': i, 'value': i} for i in range(1,11)],
                                        value=1,
                                        labelStyle={'display': 'inline-block'}
                                    ),
                                    html.Div(
                                        id="arrays analys factor",
                                        className="array array-grid",
                                        children=[
                                            html.Div(
                                                id='modalities_coordo_negative_1', 
                                            ),
                                            html.Div(
                                                id='modalities_coordo_positive_1',
                                            ),
                                            html.Div(
                                                id='modalities_contribution_1', 
                                            ),     
                                        ]
                                    ),
                                    
                                    html.P(
                                        id='sentence_negative',
                                        className="answers"
                                    ),
                                    html.P(
                                        id='sentence_positive',
                                        className="answers"
                                    ),
                                    html.Div(
                                        id='graph_patient_feature',
                                        children=[
                                            html.H4(
                                                children="Graphe des coordonnées des patients dans le plan des facteurs de l'ACM selon la variable"
                                            ),
                                            html.P(children='Choisissez le facteur pour l\'axe des abscisses', className="question"),
                                            dcc.RadioItems(
                                                id='factor_abs_graphe_pat_var',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=1,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children='Choisissez le facteur pour l\'axe des ordonnées', className="question"),
                                            dcc.RadioItems(
                                                id='factor_ordo_graphe_pat_var',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=2,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Choisissez une variable", className="question"),
                                            dcc.Dropdown(
                                                id="choose_feature",
                                                options=[{'label':str(i) ,'value':str(i)} for i in self.df_data.columns],
                                                value='dependance'
                                            ),
                                            dcc.Graph(id='patients_var', config ={'autosizable' :True, 'responsive':False})      
                                        ]
                                    )
                                ]
                            ),
                            html.Div(id="2e graphe ACM",
                                children=[
                                    html.H3(children="Graphe des coordonnées des patients sur le plan des facteurs de l'ACM "),
                                    html.Div(
                                        id = 'graph_patients',
                                        children=[
                                            html.P(children="Choisissez le facteur pour l'axe des abscisses", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id1_p',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=1,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Choisissez le facteur pour l'axe des ordonnées", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id2_p',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=2,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Classe des patients", className="question"),
                                            dcc.Checklist(
                                                id='class_patient_2D',
                                                options=[
                                                    {'label':'pas de risque', 'value':1},
                                                    {'label':'a risque', 'value':3},
                                                    {'label':'psychose', 'value':5}
                                                ],
                                                value=[1,3,5],
                                                labelStyle={'display':'inline-block'}
                                            ),
                                            html.P(children="Choississez les patients que vous voulez afficher", className="question"),
                                            dcc.Dropdown(
                                                id='choose_pat_2d',
                                                options=[{'label':i, 'value':i} for i in range (self.table_patients_mca.shape[1])],
                                                value=[i for i in range (self.table_patients_mca.shape[1])],
                                                multi=True
                                            ),
                                            html.Button(
                                                id='submit-button-state_1', 
                                                n_clicks=0, 
                                                children='Afficher le graphe',
                                                style={'margin-top':'40px'}
                                            ),
                                            html.Hr(),
                                            dcc.Loading(
                                                id="loading-1",
                                                type="default",
                                                children=dcc.Graph(id='patients_2d',
                                                    config ={'autosizable' :True, 'responsive':False},
                                                    className="margin-30")
                                            )
                                        ]

                                    )
                                ]
                            ),
                            html.Div(id="3e graphe ACM : patients 3D",
                                children=[
                                    html.H3(children="Graphe des coordonnées des patients sur le plan en 3D des facteur de l'ACM"),
                                    html.Div(
                                        id='graph3D_patients',
                                        children=[
                                            html.P(children="Choisissez le facteur pour l'axe des abscisses", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id1_p_3D',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=1,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Choisissez le facteur pour l'axe des ordonnées'", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id2_p_3D',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=2,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Choisissez le facteur pour l'axe z", className="question"),
                                            dcc.RadioItems(
                                                id='fs_id3_p_3D',
                                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                                value=3,
                                                labelStyle={'display': 'inline-block'}
                                            ),
                                            html.P(children="Classe des patients", className="question"),
                                            dcc.Checklist(
                                                id='class_patient_3D',
                                                options=[
                                                    {'label':'pas de risque', 'value':1},
                                                    {'label':'a risque', 'value':3},
                                                    {'label':'psychose', 'value':5}],
                                                value=[1,3,5],
                                                labelStyle={'display':'inline-block'}
                                            ),
                                            dcc.Dropdown(
                                                id='list_patients_keep_3d',
                                                options=[{'label':i, 'value':i} for i in range (self.table_patients_mca.shape[1])],
                                                value=[i for i in range (self.table_patients_mca.shape[1])],
                                                multi=True
                                            ),
                                            html.Button(
                                                id='submit-button-state_2', 
                                                n_clicks=0, 
                                                children='Afficher le graphe',
                                                style={'margin-top':'40px'}
                                            ),
                                            html.Hr(),
                                            dcc.Loading(
                                                id="loading-2",
                                                type="default",
                                                children=dcc.Graph(id='patients_3D',
                                                    config ={'autosizable' :True, 'responsive':False},
                                                    className="margin-30")
                                            ),
                                            html.P(children="Si le patients a une coordonnée positive selon le facteur en abscisse, il aura tendance à avoir : "),
                                            html.P(id='contributions_fs_x'),
                                            html.P(children="Si le patients a une coordonnée négative selon le facteur en abscisse, il aura tendance à avoir : "),
                                            html.P(id='contributions_fs_x_')    
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(id="all the ml",
                        className="new-section",
                        children=[
                            html.H2(children="Machine learning"),
                            html.Div(id="Data preparation",
                                children=[
                                    html.H3(children="Preprocessing"),
                                    html.P(children='Quel référentiel de temps voulez vous choisir pour le diagnostic du patient ?', className="question"),
                                    dcc.RadioItems(
                                        id='choose_period_decision_tree',
                                        options=[
                                            {'label':'M0', 'value':0},
                                            {'label':'M6', 'value':1},
                                            {'label':'M12', 'value':2},
                                            {'label':'M18', 'value':3},
                                            {'label':'M24', 'value':4}
                                        ],
                                        value=4,
                                        labelStyle={'display' : 'inline-block'}
                                    ),
                                    html.P(children="Choississez le pourcentage des données utilisées pour tester l'arbre de décision. \
                                                    Le reste des données sera utilisé pour entrainer l'arbre ", className="question"
                                    ),
                                    dcc.RadioItems(
                                        id='split_size',
                                        options=[
                                            {'label':'10%', 'value':0.1},
                                            {'label':'20%', 'value':0.2},
                                            {'label':'30%', 'value':0.3},
                                            {'label':'40%', 'value':0.4}
                                        ],
                                        value=0.3,
                                        labelStyle={'display' : 'inline-block'}
                                    ),
                                    html.P(children="Voulez-vous garder les patients atteints de psychose de l'étude ?", className="question"),
                                    dcc.RadioItems(
                                        id='keep_psychose',
                                        options=[
                                            {'label':'oui', 'value': 1},
                                            {'label':'non', 'value': 0}
                                        ],
                                        value=1,
                                        labelStyle={'display': 'inline-block'}
                                    ),
                                    html.P('Voulez-vous mettre un patient de côté pour faire une prédiction de sa classe ?', className="question"),  
                                    dcc.RadioItems(
                                        id="keep_patient_aside",
                                        options=[
                                            {'label':'oui', 'value': 1},
                                            {'label':'non', 'value': 0}
                                        ],
                                        value = 1,
                                        labelStyle={"display":"inline-block"}
                                    ),
                                    html.Div(id="display_patient_to_choose",
                                        children=[
                                            html.P("Choisissez le patient que vous voulez mettre de côté pour ensuite prédire sa classe", className="question"),
                                            dcc.Dropdown(
                                                id='patients_to_evaluate',
                                                options=[{'label':i, 'value':i} for i in (self.df_data_disj.index.to_list())],
                                                value=None
                                            ), 
                                            html.Div(id="hidden_div5", style={'display':'none'}),
                                        ]
                                    ),
                                    html.P(id='data_preparation_state', children='0', style={'opacity': '0'}),                
                                ]
                            ),
                            html.Div(id="all of the decision tree",
                                className="new-section",
                                children=[
                                    html.Div(id="all the parameters of the tree",
                                        children=[
                                            html.H2(
                                                children="Modèle de ML : arbre de décision"
                                            ),
                                            html.Div(id="div for the parameters of the tree",
                                                children=[
                                                    html.H4(children="Calcul des meilleurs paramètres"),
                                                    html.P(children="Choissez la métrique pour calculer les meilleurs paramètres",
                                                        className="question"
                                                    ),
                                                    dcc.RadioItems(
                                                        id="scoring_best_para",
                                                        options=[
                                                            {'label': 'accuracy', 'value': 'accuracy'},
                                                            {'label':'f1_macro', 'value': 'f1_macro'}
                                                        ],
                                                        value ="accuracy",
                                                        labelStyle={'display' : 'inline-block'}
                                                    ),

                                                    html.P(children="Les paramètres meilleurs paramètres pour l'arbre de décision sont :",
                                                        className="answers"
                                                    ),
                                                    html.P(id='best_para_tree'),

                                                    html.H4(
                                                        children="Paramétrage du modèle",
                                                    ),  
                                                    html.P(
                                                        children="Profondeur maximale de l'arbre :",
                                                        className="question"
                                                    ),
                                                    dcc.RadioItems(
                                                        id='depth_',
                                                        options=[{'label':i, 'value': i} for i in [1,2,3,4,5]],
                                                        value=3,
                                                        labelStyle={"display":"inline-block"}
                                                    ),
                                                    html.P(children="Poids des classes :", className="question"),
                                                    html.Div(
                                                        id="div_weight_with_psychose",
                                                        children=[
                                                            dcc.RadioItems(
                                                                id='weight_with_psychose',
                                                                options=[
                                                                    {'label': "pas de poids", 'value' : str({'1': 1,  '3': 1, '5': 1})},
                                                                    {'label': "classe 'pas de risque' accentuée", 'value' : str({'1': 10,  '3': 1, '5': 1})},
                                                                    {'label': "classe 'a risque' accentuée", 'value' : str({'1': 1,  '3': 10, '5': 1})},
                                                                    {'label': "classe 'psychose' accentuée", 'value' : str({'1': 1,  '3': 1, '5': 10})},
                                                                ],
                                                                value=str({'1': 1,  '3': 1, '5': 1}),
                                                                labelStyle={"display":"inline-block"}
                                                            )
                                                        ]
                                                    ),
                                                    html.Div(
                                                        id="div_weight_without_psychose",
                                                        children=[
                                                            dcc.RadioItems(
                                                                id='weight_without_psychose',
                                                                options=[
                                                                    {'label': "pas de poids", 'value' : str({'1': 1,  '3': 1})},
                                                                    {'label': "classe 'pas de risque' accentuée", 'value' : str({'1': 10,  '3': 1})},
                                                                    {'label': "classe 'a risque' accentuée", 'value' : str({'1': 1,  '3': 10})}
                                                                ],
                                                                value=str({'1': 1,  '3': 1}),
                                                                labelStyle={"display":"inline-block"}
                                                            )
                                                        ]
                                                    ),
                                                    html.P(
                                                        children="Nombre minimal de patients requis à chaque noeud :",
                                                        className="question"
                                                    ),
                                                    dcc.Slider(
                                                        id='min_split_',
                                                        min=2,
                                                        max=10,
                                                        marks={i: 'min_split {}'.format(i) for i in range(2,11)},
                                                        value=5
                                                    ),
                                                    html.P(children="Nombre minimal de patients requis à chaque feuille :",
                                                        className="question"
                                                    ),
                                                    dcc.Slider(
                                                        id='min_leaf_',
                                                        min=1,
                                                        max=10,
                                                        marks={i: 'min_leaf {}'.format(i) for i in range(1,11)},
                                                        value=2
                                                    ),
                                                    html.P(children="Variable(s) à ne pas prendre en compte pour l'arbre :",
                                                        className="question"
                                                    ),
                                                    dcc.Dropdown(
                                                        id='delete_col',
                                                        options=[{'label': i, 'value': i} for i in self.x_train.columns.get_level_values('features').unique().to_list()],
                                                        multi=True,
                                                        value="False"
                                                    ) 
                                                ]
                                            )    
                                        ]
                                    ),
                                    html.H4(children="Arbre de décision"),
                                    html.Div(className="center-img",
                                        children=[
                                            html.Img(id='decision tree', className="decision-tree")
                                        ]
                                    ),
                                    html.P(
                                        id="explication_arbre", className="explanation",
                                        children="L'arbre de décision essaie de séparer les différentes classes 'pas de risque', \
                                            'a risque' et 'psychose' à chaque noeud. L'objectif est de trouver des combinaisons de variables permettant d'avoir des feuilles \
                                            les plus pures possible, c'est à dire ayant uniquement des patients de la même classe. "
                                    ),
                                    html.Div(id='analysis prediction tree',
                                        children=[
                                            html.H4(children="Etude de l'arbre de décision"),
                                            html.H5(id="confusion matrix title", children="Matrice de confusion de l'arbre de décision"),
                                            html.P(id="explication_tree", children="Les lignes de la matrice de confusion représentent les classes\
                                                de l'étude et la somme de chaque ligne le nombre d'éléments pour la classe. \
                                                Les colonnes représentent les classes prédites. Idéalement, tous les éléments se retrouvent sur la \
                                                diagonales : ils ont été prédits dans leur classe.", className="explanation"
                                            ),
                                            html.Div(id="confusion matrix",
                                                className="confusion-matrix",
                                                children=[
                                                    html.Div(id="column", children=["Prédictions"]),
                                                    html.Div(id="confu_matrix_1"),
                                                    html.Div(id="confu_matrix_2"),
                                                    html.Div(id="confu_matrix_3")
                                                ]
                                            ),
                                            html.Div(id="classi_report",
                                                children=[
                                                    html.H5(id="classifiaction report title", children="Analyse des résultats de la prédiction"),
                                                    html.Div(id="classification report",
                                                        style={'display':'grid',"margin-right":"70%","font-size":"15px",
                                                                "text-align-last": "right","white-space":"pre"},
                                                        children=[
                                                            html.P(id="classi_rep_1"),
                                                            html.P(id="classi_rep_2"),
                                                            html.P(id="classi_rep_3"),
                                                            html.P(id="classi_rep_4"),
                                                            html.P(id="classi_rep_5"),
                                                            html.P(id="classi_rep_6"),
                                                            html.P(id="classi_rep_7"),
                                                            html.P(id="classi_rep_8"),
                                                            html.P(id="classi_rep_9"),
                                                            html.P(id="classi_rep_10")
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Div(id="multilabel_confusion_matrix",
                                                children=[
                                                    html.H5(id="mutlilabel title", children="Matrice de confusion pour chaque classe :"),
                                                    html.P(className="explanation",
                                                        children="Nous réalisons ici la matrice de confusion pour chaque classe, indépendamment des autres. C'est à dire que nous testons si le patient appartient ou non à la classe. \
                                                            Nous avons horizontalement, sur la première ligne les patients qui n'appartiennent pas à la classe en question et sur la seconde ligne, les patients qui appartiennet à la classe. \
                                                            Verticalement, nous avons les prédictions. Dans la première colonne, on a les patients qui n'ont pas été prédits comme appartenant à la classe et dans la seconde colonne ceux ayant \
                                                            été prédits comme appartenant à la classe. On voit donc que plus le modèle est performant, plus il y a de patients sur la diagonale, la première case \
                                                            représentant les vrais négatifs et celle en bas à droite les vrais positifs."
                                                    ),
                                                    html.Div(id="mutlilabel_confu_mat",
                                                        style={'display':'grid',"margin-right":"70%","font-size":"15px",
                                                                "text-align-last": "right","white-space":"pre"
                                                        },
                                                        children=[
                                                            html.P(id="multi_confu_1"),
                                                            html.P(id="multi_confu_2"),
                                                            html.P(id="multi_confu_3"),
                                                            html.P(id="multi_confu_4"),
                                                            html.P(id="multi_confu_5"),
                                                            html.P(id="multi_confu_6"),
                                                            html.P(id="multi_confu_7"),
                                                            html.P(id="multi_confu_8")
                                                                    
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            html.Div(id="all of the random forest", 
                                className="new-section",
                                children=[
                                    html.Div(id="preprocessing and explanation of random forest",
                                        children=[
                                            html.H2(children="Random forest"),
                                            html.P(id="explication_rf_all",
                                                className="explanation",
                                                children=" La forêt aléatoire est une technique d'apprentissage supervisé \
                                                    basée sur l'utilisation de nombreux arbres de décision. Chaque arbre prend seulement une partie des données \
                                                    pour s'entrainer, différente entre chaque arbre. L'autre partie de données est utilisée pour le tester. Ainsi, \
                                                    chaque arbre sera entrainé et testé sur des données différentes. Par ailleurs, chaque arbre a un nombre limité \
                                                    de modalités disponibles pour faire les tests de séparation. Il choisira a chaque noeud la meilleure parmi celles à \
                                                    disposition. Cela permet de forcer l'utilisation de différentes modalités et de tester différents enchainement de tests."
                                            ),
                                            html.Div(id="hidden_div3", style={"display":"none"}),
                                            html.Div(id="best_param_random_forest", 
                                                children=[
                                                    html.H4("Calcul des meilleurs paramètres"),   
                                                    html.P("Sur quel scoring voulez-vous calculer ces paramètres (accuracy, f1_score):", className="question"), 
                                                    dcc.Input(
                                                        id='scoring_best_param_random_forest',
                                                        value='accuracy', 
                                                        type='text'
                                                    ),
                                                    html.P("Voici le meilleur paramétrage :"),    
                                                    html.P(id = "best_param_random_forest_display")
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(id="all the parameters of the rf",
                                        children=[
                                            html.H4(children="Paramétrage du modèle"),
                                            html.P("Voulez-vous choisir les meilleurs paramètres pour la forêt aléatoire ? ", className="question"),  
                                            dcc.RadioItems(
                                                id="use_best_param",
                                                options=[
                                                    {'label':'non', 'value': 0},
                                                    {'label':'oui', 'value': 1}
                                                ],
                                                value=0,
                                                labelStyle={"display":"inline-block"}
                                            ),
                                            html.P("Profondeur maximale de l'arbre :", className="question"),
                                            dcc.RadioItems(
                                                id='max_depth_rf',
                                                options=[{'label':i, 'value': i} for i in [2,3,4,'none']],
                                                value=3,
                                                labelStyle={"display":"inline-block"}
                                            ),
                                            html.P("Nombre d'arbres de décision dans la forêt aléatoire :", className="question"),
                                            dcc.RadioItems(
                                                id='n_estimators',
                                                options=[
                                                    {'label':10, 'value': 10},
                                                    {'label':20, 'value': 20},
                                                    {'label':50, 'value': 50},
                                                    {'label':100, 'value': 100}
                                                ],
                                                value=50,
                                                labelStyle={"display":"inline-block"}
                                            ),
                                            html.P("Nombre maximal de modalités disponibles pour un arbre :", className="question"),
                                            dcc.RadioItems(
                                                id='max_features_rf',
                                                options=[
                                                    {'label':10, 'value': 10},
                                                    {'label':30, 'value': 30},
                                                    {'label':50, 'value': 50},
                                                    {'label':70, 'value': 70},
                                                    {'label':100, 'value': 100}
                                                ],
                                                value=30,
                                                labelStyle={"display":"inline-block"}
                                            ),
                                            html.P("Nombre minimum de patients en sortie de noeud ", className="question"),
                                            dcc.Slider(
                                                id='min_samples_split_rf',
                                                min=2,
                                                max=10,
                                                marks={i: 'min_split {}'.format(i) for i in range(2,11)},
                                                value=5
                                            ),
                                            html.P("Nombre minimum de patients dans une feuille ", className="question"), 
                                            dcc.Slider(
                                                id='min_samples_leaf_rf',
                                                min=1,
                                                max=10,
                                                marks={i: 'min_leaf {}'.format(i) for i in range(1,11)},
                                                value=5
                                            ),
                                            html.P("Voulez-vous retirer des variables du modèle ? ", className="question"), 
                                            dcc.Dropdown(
                                                id='delete_col_rf',
                                                options=[{'label': i, 'value': i} for i in self.x_train.columns.get_level_values('features').unique().to_list()],
                                                multi=True,
                                                value=[]
                                            ) 
                                        ]
                                    ),           
                                    html.Div(id='analysis random forest',
                                        children=[
                                            html.H4(children="Etude de la forêt aléatoire"),
                                            html.H5("Résultats de la prédiction :"),
                                            dcc.Loading(
                                                id="loading-6",
                                                type="default",
                                                children=html.P(id="results_prediction_rf", style={"margin-bottom":"15px"})
                                            ),
                                            html.H5(id='list_feature_importances_title', children="Importance des variables dans la forêt aléatoire "),
                                            html.Div(id="list_feature_importances", className="rf-feature-importance"),
                                            html.H5(id="confusion matrix rf title", children="Matrice de confusion de la forêt aléatoire"),
                                            html.P(id="explication_rf", 
                                                children="Les lignes de la matrice de confusion représentent les classes\
                                                    de l'étude et la somme de chaque ligne le nombre d'éléments pour la classe. \
                                                    Les colonnes représentent les classes prédites. Idéalement, tous les éléments se retrouvent sur la \
                                                    diagonales : ils ont été prédits dans leur classe.",
                                                style={"margin-bottom":"20px"}
                                            ),
                                            html.Div(id="confusion matrix rf",
                                                className="confusion-matrix",
                                                children=[
                                                    html.P(id="column_rf", children=["Prédictions"]),
                                                    html.P(id="confu_matrix_rf_1"),
                                                    html.P(id="confu_matrix_rf_2"),
                                                    html.P(id="confu_matrix_rf_3")
                                                ]
                                            ),
                                            html.Div(id="classi_report_rf",
                                                children=[
                                                    html.H5(id="classification report title", children="Analyse de la forêt aléatoire"),
                                                    html.Div(id="classification report_rf",
                                                        style={'display':'grid',"margin-right":"70%","font-size":"15px",
                                                            "text-align-last": "right","white-space":"pre"},
                                                        children=[
                                                            html.P(id="classi_report_rf_1"),
                                                            html.P(id="classi_report_rf_2"),
                                                            html.P(id="classi_report_rf_3"),
                                                            html.P(id="classi_report_rf_4"),
                                                            html.P(id="classi_report_rf_5"),
                                                            html.P(id="classi_report_rf_6"),
                                                            html.P(id="classi_report_rf_7"),
                                                            html.P(id="classi_report_rf_8"),
                                                            html.P(id="classi_report_rf_9"),
                                                            html.P(id="classi_report_rf_10")
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Div(id="multilabel_confusion_matrix_rf",
                                                children=[
                                                    html.H5(id="mutlilabel title rf", children="Matrice de confusion pour chaque classe"),
                                                    html.Div(id="mutlilabel_confu_mat_rf",
                                                        style={'display':'grid',"margin-right":"70%","font-size":"15px",
                                                            "text-align-last": "right","white-space":"pre"},
                                                        children=[
                                                            html.P(id="multilab_confu_mat_rf_1"),
                                                            html.P(id="multilab_confu_mat_rf_2"),
                                                            html.P(id="multilab_confu_mat_rf_3"),
                                                            html.P(id="multilab_confu_mat_rf_4"),
                                                            html.P(id="multilab_confu_mat_rf_5"),
                                                            html.P(id="multilab_confu_mat_rf_6"),
                                                            html.P(id="multilab_confu_mat_rf_7"),
                                                            html.P(id="multilab_confu_mat_rf_8")
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ]
                                    )    
                                ]
                            )
                        ]
                    )
                ],
            )   
        ]

        # Callbacks

        # File to process
        self.app.callback(
            Output('hidden_div','children'),
            [Input('name_file', 'value'),
            Input('option_lost', 'value')]
        )(self.choose_patients_lost)

        # Chi2
        self.app.callback(
            [Output('list of correlation1', 'children'),
            Output('list of correlation2', 'children'),
            Output('list of correlation3', 'children')],
            [Input('threshold_chi2', 'value'),
            Input('minimum_elements', 'value'),
            Input('chi2_label', 'value')]
        )(self.display_list_chi2_correlation)
    
        # Intro MCA
        self.app.callback(
            Output('effective_modality_display', 'children'),
            [Input('effective_modality','value'),
            Input('choose_period_graph', 'value')]
        )(self.display_modality_effective)

        # Var by var graph
        self.app.callback(
            Output('var_by_var', "figure"),
            [Input('submit-button-state_3', 'n_clicks')],
            [State("fs_id1_", "value"),
            State("fs_id2_", "value"),
            State('choose_period_graph', 'value'),
            State('significant_only', 'value')]
        )(self.display_graph_var_by_var)

        # Explication graph var by var
        self.app.callback(
            Output("modality_to_evaluate",'options'),
            [Input("fs_id1_", "value"),
            Input("fs_id2_", "value")]
        )(self.display_modalities_options)

        self.app.callback(
            Output("hidden_div4", 'children'),
            [Input("fs_id1_", "value"),
            Input("fs_id2_", "value")]
        )(self.calculate_df_distance)

        self.app.callback(
            [Output('array_graph_moda', 'children'),
            Output("text_before_array", 'children')],
            [Input("dist_moda", "value"),
            Input("modality_to_evaluate", 'value')]
        )(self.explanation_graph_modalities)

        # Modalities contribution to the factors
        self.app.callback(
            [Output('modalities_coordo_positive_1', 'children' ),
            Output('modalities_coordo_negative_1', 'children'),
            Output('modalities_contribution_1', 'children'),
            Output('sentence_positive', 'children'),
            Output('sentence_negative', 'children')],
            [Input('factor_to_analyse','value')]
        )(self.display_modalities_factor)

        # Graph patient/feature
        self.app.callback(
            Output('patients_var', 'figure'),
            [Input('factor_abs_graphe_pat_var', 'value'),
            Input('factor_ordo_graphe_pat_var', 'value'),
            Input('choose_feature', 'value'),
            Input('choose_period_graph', 'value')]
        )(self.create_graph_pat_feature)

        # Patients 2D graph
        self.app.callback(
            Output('patients_2d', "figure"),
            [Input('submit-button-state_1', 'n_clicks')],

            [State("fs_id1_p", "value"),
            State("fs_id2_p", "value"),
            State("class_patient_2D","value"),
            State("choose_pat_2d","value")]
        )(self.display_graph_patients)

        # Patients 3D graph
        self.app.callback(
            [Output('patients_3D', "figure"),
            Output('contributions_fs_x','children'),
            Output('contributions_fs_x_','children')
            ],
            [Input('submit-button-state_2', 'n_clicks')],
            [State("fs_id1_p_3D", "value"),
            State("fs_id2_p_3D", "value"),
            State("fs_id3_p_3D", "value"),
            State("class_patient_3D", "value"),
            State("list_patients_keep_3d", "value")]
        )(self.graph_3D_patients)

        # Machine learning

        # Create train and test data for ML tasks
        self.app.callback(
            Output('display_patient_to_choose','style'),
            [Input('keep_patient_aside','value')]
        )(self.put_pat_aside)

        self.app.callback(
            [Output('hidden_div5', 'children'),
            Output('data_preparation_state','children'),
            Output('div_weight_with_psychose', 'style'),
            Output('div_weight_without_psychose', 'style')],
            [Input('patients_to_evaluate', 'value'),
            Input('choose_period_decision_tree', 'value'),
            Input('split_size', 'value'),
            Input('keep_psychose', 'value')]
        )(self.update_data)

        # Decision Tree

        # Best parameters tree
        self.app.callback(
            Output('best_para_tree', 'children'),
            [Input('scoring_best_para', 'value')]
        )(self.best_para_tree)

        # decision tree
        self.app.callback(
            [
                Output('decision tree', 'src'),
                Output('confu_matrix_1', 'children'),
                Output('confu_matrix_2', 'children'),
                Output('confu_matrix_3', 'children'),
                Output('classi_rep_1', 'children'),
                Output('classi_rep_2', 'children'),
                Output('classi_rep_3', 'children'),
                Output('classi_rep_4', 'children'),
                Output('classi_rep_5', 'children'),
                Output('classi_rep_6', 'children'),
                Output('classi_rep_7', 'children'),
                Output('classi_rep_8', 'children'),
                Output('classi_rep_9', 'children'),
                Output('classi_rep_10', 'children'),
                Output('multi_confu_1', 'children'),
                Output('multi_confu_2', 'children'),
                Output('multi_confu_3', 'children'),
                Output('multi_confu_4', 'children'),
                Output('multi_confu_5', 'children'),
                Output('multi_confu_6', 'children'),
                Output('multi_confu_7', 'children'),
                Output('multi_confu_8', 'children')
            ],
            [
                Input('depth_', 'value'),
                Input('min_split_', 'value'),
                Input('min_leaf_', 'value'),
                Input('weight_with_psychose','value'),
                Input('weight_without_psychose','value'),
                Input('delete_col','value'),
                Input('data_preparation_state', 'children')
            ],
            [
                State('keep_psychose', 'value'),
                State('choose_period_decision_tree', 'value'),
                State('patients_to_evaluate', 'value'),
                State('split_size', 'value'),
            ]
        )(self.update_graph)

        # Random forest

        # Best param
        self.app.callback(
            Output('best_param_random_forest_display', 'children'),
            [Input('scoring_best_param_random_forest', 'value')]
        )(self.best_para_rf)

        # Process random forest
        self.app.callback(
            [
                Output('list_feature_importances','children'),
                Output('confu_matrix_rf_1', 'children'),
                Output('confu_matrix_rf_2', 'children'),
                Output('confu_matrix_rf_3', 'children'),
                Output('classi_report_rf_1','children'),
                Output('classi_report_rf_2','children'),
                Output('classi_report_rf_3','children'),
                Output('classi_report_rf_4','children'),
                Output('classi_report_rf_5','children'),
                Output('classi_report_rf_6','children'),
                Output('classi_report_rf_7','children'),
                Output('classi_report_rf_8','children'),
                Output('classi_report_rf_9','children'),
                Output('classi_report_rf_10','children'),
                Output('multilab_confu_mat_rf_1','children'),
                Output('multilab_confu_mat_rf_2','children'),
                Output('multilab_confu_mat_rf_3','children'),
                Output('multilab_confu_mat_rf_4','children'),
                Output('multilab_confu_mat_rf_5','children'),
                Output('multilab_confu_mat_rf_6','children'),
                Output('multilab_confu_mat_rf_7','children'),
                Output('multilab_confu_mat_rf_8','children'),
                Output('results_prediction_rf', 'children')
            ],
            [
                Input('max_depth_rf', 'value'),
                Input('n_estimators','value'),
                Input('max_features_rf', 'value'),
                Input('min_samples_split_rf', 'value'),
                Input('min_samples_leaf_rf', 'value'),
                Input('delete_col_rf','value'),
                Input('use_best_param','value'),
                Input('data_preparation_state', 'children'),
            ],
            [
                State('patients_to_evaluate', 'value'),
                State('choose_period_decision_tree', 'value'),
                State('split_size', 'value'),
                State('keep_psychose', 'value'),
            ]    
        )(self.random_forest_app)
            




    def choose_patients_lost(self, name_file, option_lost):
        self.df_data, self.df_label = pipeline_preprocessing(name_file, option_lost, 0)
        return ''
    
    def display_modality_effective(self, modality, period):
        try :
            modality_ = convert_tring_to_tuple(modality)
            effective = self.df_data_disj[modality_[0],modality_[1]].sum(axis=0)
            if effective>1: #patient plurial
                return('L\'effectif de la modalité {} est de {} patients'.format(str(modality), str(effective)))
            else:
                return('L\'effectif de la modalité {} est de {} patient'.format(str(modality), str(effective)))
        except:
            return 'Veuillez choisir une modalité'
    
    def display_graph_var_by_var(self, n_clicks, fs_id1_, fs_id2_, period_for_var, significant_only):
        df_var_period = self.list_df_data_disj[period_for_var]
        data, layout = interactive_plot_variable_by_variable(self.table_modalities_mca, self.table_explained_mca, fs_id1_, fs_id2_,  False, True, significant_only)    
        return {'data': data,'layout': layout }
    
    def display_modalities_options(self, fs_id1, fs_id2):
        list_modalities = []
        for i, modality in enumerate(self.table_modalities_mca.columns):
            if abs(self.table_modalities_mca.loc[('Factor', fs_id1), modality]) > math.sqrt(self.table_explained_mca.loc[fs_id1,'Zλ']) and \
                abs(self.table_modalities_mca.loc[('Factor', fs_id2), modality]) > math.sqrt(self.table_explained_mca.loc[fs_id2,'Zλ']):
                list_modalities.append(modality)

        return [{'label': str(i) ,'value': str(i)} for i in list_modalities]  #TODO value (string | number | list of string | numbers; optional)

    def calculate_df_distance(self, fs_id1, fs_id2):
        vect, pos_x, pos_y, new_norm = position_vector(self.table_modalities_mca, fs_id1, fs_id2)
        self.df_distances = creation_dataframe_distance_modalities(pos_x, pos_y, fs_id1, fs_id2, self.table_modalities_mca, self.table_explained_mca)        
        return ''

    def explanation_graph_modalities(self, dist_moda, modality):
        """
        Display the modalities in the ACM graph which have a distance lower than dist_moda with the modality 
        """
        dist_moda = dist_moda/10  # The distance was an integer, resize it as decimal
        if isinstance(modality, str):
            modality = convert_tring_to_tuple(modality)

        # If df_distances exists
        if list(self.df_distances.columns):
            df_closest_moda = select_dist_modalities(modality, self.df_distances, dist_moda, self.df_data_disj, self.df_label_disj)
    
            if df_closest_moda.empty:
                return(['Il n\'y a pas de modalités proches'], ["Voici les modalités dans le rayon de la modalité \"{}: {} \"".format(str(modality[0]),str(modality[1]))])
            elif df_closest_moda.empty == False:
                return([generate_table(df_closest_moda)], ["Voici les modalités dans le rayon de la modalité \"{}: {} \"".format(str(modality[0]),str(modality[1]))])
        else:
            return "", ""

    def display_modalities_factor(self, factor_to_analyse):
        """
        Analyse a factor of the ACM, listing the modalities which contribute the max positively and negatively
        and the absolute contribution
        """
        df_positive_coordo = select_max_positive_contribution_fs(self.table_modalities_mca, factor_to_analyse, 5, 'coordonnées')
        df_negative_coordo = select_max_negative_contribution_fs(self.table_modalities_mca, factor_to_analyse, 5, 'coordonnées')
        df_contribution = select_max_positive_contribution_fs(self.table_modalities_mca_contribution, factor_to_analyse, 10, 'contribution x1000')
        
        sentence_positive, sentence_negative = translate_contribution_to_sentence(df_contribution, self.table_modalities_mca, factor_to_analyse)
        return (generate_table(df_positive_coordo), generate_table(df_negative_coordo), generate_table(df_contribution),
            sentence_positive, sentence_negative )

    def create_graph_pat_feature(self, fs_1, fs_2, feature, period):
        """
        Display the patients on the selected factors, colored according to their belonging to the modalities
        of the feature
        """
        data, layout = interactive_plot_patient_modality(feature,
                                                        self.df_data_disj,
                                                        self.list_table_patients_mca_time[period],
                                                        self.table_modalities_mca,
                                                        self.df_data,
                                                        fs_id1=fs_1,
                                                        fs_id2=fs_2,
                                                        display=False)
        return {'data': data,'layout': layout }

    def display_graph_patients(self, n_clicks,fs_id1_p, fs_id2_p, class_patient_2D, choose_pat_2d):
        """
        Graph of the patients in the MCA factor plan, colored depending on their label
        """
        data, layout = interactive_plot_patient_time(df=self.table_patients_mca,
                                                    fs_id1=fs_id1_p,
                                                    fs_id2=fs_id2_p,
                                                    df_label_color=self.df_color_all_lab,
                                                    class_patient=class_patient_2D,
                                                    display=False,
                                                    list_patients_to_keep=choose_pat_2d)
        return {'data': data,'layout': layout }
    
    def graph_3D_patients(self, n_clicks_2, fs_id1_p_3D, fs_id2_p_3D, fs_id3_p_3D, class_patient_3D, list_patients_keep_3d):
        data, layout = interactive_plot_patient_time_follow_3d(list_df=self.list_table_patients_mca_time, 
                                                               df_label_color=self.df_color_all_lab, 
                                                               fs_id1=fs_id1_p_3D,
                                                               fs_id2=fs_id2_p_3D,
                                                               fs_id3=fs_id3_p_3D,
                                                               class_patient=class_patient_3D,
                                                               display=False,
                                                               list_patients_to_keep=list_patients_keep_3d)
        
        contri_val_x, contri_col_x = select_max_positive_contribution_fs(self.table_modalities_mca, fs_id1_p_3D, 5, 'coordonnées')  #reprendre la table affichée
        contri_val_x_, contri_col_x_ = select_max_negative_contribution_fs(self.table_modalities_mca, fs_id1_p_3D, 5,'coordonnées')

        return {'data': data,'layout': layout }, str(contri_col_x), str(contri_col_x_)

    def put_pat_aside (self, keep_pat_aside):
            if keep_pat_aside==False:
                return({'display':'none'} )
            else:
                return {'margin-top':'10px'} 

    def update_data (self, patient, choose_period_decision_tree, split_size, keep_psychose):
        """
        Create 2 dataframes for the machine learning tasks, df_data_disj_ml and df_label_ml, copies of f_data_disj and df_label.
        These copies are split into x_train, y_train, x_test, y_test later for the ml tasks.
        This allows us to keep the integrity of the df_data_disj as we can drop some rows.

        It also removes a row corresponding to the id "patient" in the data. This is used to isolate a patient from the training and later
        test a prediction on it.
        """
        self.df_patient_pred = pd.DataFrame(0, index=[0], columns=self.df_data_disj.columns)
        self.df_patient_pred_label = pd.DataFrame()
        self.df_data_disj_ml = deepcopy(self.df_data_disj)
        self.df_label_ml = deepcopy(self.df_label)
        
        if patient:
            patient = int(patient)
            # Pick the patient to keep aside for prediction in another dataframe
            self.df_patient_pred = self.df_patient_pred.append(self.df_data_disj_ml.loc[patient])
            self.df_data_disj_ml = self.df_data_disj_ml.drop([patient], axis=0)
            self.df_data_disj_ml.index = [i for i in range (self.df_data_disj_ml.shape[0])]  # Re-index
            # Same for the label
            self.df_patient_pred_label = self.df_patient_pred_label.append(self.df_label_ml.loc[patient])
            self.df_label_ml = self.df_label_ml.drop([patient], axis=0)
            self.df_label_ml.index = [i for i in range (self.df_label_ml.shape[0])]  # Re-index

        # Process the split in case there are modifications
        self.x_train, self.y_train, self.x_test, self.y_test = split_train_test(self.df_data_disj_ml,
                                                                                self.df_label_ml,
                                                                                choose_period_decision_tree,
                                                                                split_size,
                                                                                keep_psychose)
        self.counter += 1
        print("Size after changing params", len(self.x_train.index))
        if keep_psychose:
            return '', str(self.counter), {'margin_bottom':'20px'}, {'display':'none'}
        else:
            return '', str(self.counter), {'display':'none'}, {'margin_bottom':'20px'}
    
    def best_para_tree(self, scoring_best_para):
        best_score, grid_score = best_param_tree(scoring_best_para, 5, self.x_train, self.y_train)
        return str(best_score)

    def update_graph(self, depth_, min_split_, min_leaf_, weight_with_psychose,
                 weight_without_psychose, delete_col, state, keep_psychose, choose_period_decision_tree, patient_removed, split_size):
        """
        
        """
        # TODO refactor
        print("Size x_train decision tree", len(self.x_train.index))
        if keep_psychose == True:  #if the patients 'psychose' are still in the study, the weight with them needs to be used
            if isinstance(weight_with_psychose, str) and weight_with_psychose[0] =="{":
                weight_with_psychose = transform_dict_weight(weight_with_psychose)

            tree, confu_matrix, classi_report, multilab_confu_mat  = plot_tree (self.x_train
                                                                            , self.y_train
                                                                            , self.x_test
                                                                            , self.y_test
                                                                            , self.classes_name_with_psy
                                                                            , depth= depth_
                                                                            , min_split= min_split_
                                                                            , min_leaf= min_leaf_
                                                                            , weight = weight_with_psychose
                                                                            , delete_col =delete_col
                                                                            )
            
            confu_mat_ok = processing_matrix_format(confu_matrix, self.classes_name_with_psy, confu_matrix=True, multilabel_confu_m=False)  # formatting the confusion matrix
            multi_confu_mat = processing_matrix_format(multilab_confu_mat, self.classes_name_with_psy, confu_matrix=False, multilabel_confu_m=True)
            encoded_image = base64.b64encode(open('./ml/images/decision_tree_modified.png', 'rb').read())
            classi_report = str(classi_report).split('\n')
            
            classi_rep_1, classi_rep_2, classi_rep_3, classi_rep_4, classi_rep_5 = classi_report[0],classi_report[1], classi_report[2], classi_report[3], classi_report[4] 
            classi_rep_6, classi_rep_7, classi_rep_8, classi_rep_9, classi_rep_10 = classi_report[5], classi_report[6], classi_report[7], classi_report[8], classi_report[9] 
        
            
            return ('data:image/png;base64,{}'.format(encoded_image.decode()),
                confu_mat_ok[0],confu_mat_ok[1],confu_mat_ok[2],classi_rep_1, classi_rep_2, classi_rep_3,
                classi_rep_4, classi_rep_5,classi_rep_6, classi_rep_7, classi_rep_8, classi_rep_9, classi_rep_10,
                multi_confu_mat[0],multi_confu_mat[1],multi_confu_mat[2],multi_confu_mat[3],multi_confu_mat[4],
                multi_confu_mat[5],multi_confu_mat[6],multi_confu_mat[7])
            
        elif keep_psychose == False:
            if isinstance(weight_without_psychose, str) and weight_without_psychose[0] =="{":
                weight_without_psychose = transform_dict_weight(weight_without_psychose)
            tree, confu_matrix, classi_report, multilab_confu_mat  = plot_tree (self.x_train
                                                                            , self.y_train
                                                                            , self.x_test
                                                                            , self.y_test
                                                                            , self.classes_name_without_psy
                                                                        , depth= depth_
                                                                        , min_split= min_split_
                                                                        , min_leaf= min_leaf_
                                                                        , weight = weight_without_psychose
                                                                        , delete_col =delete_col
                                                                        )
            
            confu_mat_ok = processing_matrix_format(confu_matrix, self.classes_name_without_psy, confu_matrix=True, multilabel_confu_m=False)  #foramtting the confusion matrix
            multi_confu_mat = processing_matrix_format(multilab_confu_mat, self.classes_name_without_psy, confu_matrix=False, multilabel_confu_m=True)

            encoded_image = base64.b64encode(open('./ml/images/decision_tree_modified.png', 'rb').read())
            classi_report = str(classi_report).split('\n')
    
            classi_rep_1, classi_rep_2, classi_rep_3, classi_rep_4, classi_rep_5 = classi_report[0],classi_report[1], classi_report[2], classi_report[3], classi_report[4] 
            classi_rep_6, classi_rep_7, classi_rep_8, classi_rep_9, classi_rep_10 = classi_report[5], classi_report[6], classi_report[7],'', ''
        
            return ('data:image/png;base64,{}'.format(encoded_image.decode()),
                confu_mat_ok[0],confu_mat_ok[1],'',classi_rep_1, classi_rep_2, classi_rep_3,
                classi_rep_4, classi_rep_5,classi_rep_6, classi_rep_7, classi_rep_8, classi_rep_9, classi_rep_10,
                multi_confu_mat[0],multi_confu_mat[1],multi_confu_mat[2],multi_confu_mat[3],multi_confu_mat[4],
                '','','')

    def best_para_rf(self, scoring_best_param_random_forest):
        grid_score_rf = random_forest_grid_search(self.x_train, self.y_train, scoring_best_param_random_forest)
        best_score_rf = grid_score_rf.best_params_
        self.best_random_forest = grid_score_rf.best_estimator_
        return(str(best_score_rf))

    def random_forest_app(self, max_depth_rf, n_estimators,
                  max_features_rf, min_samples_split_rf, min_samples_leaf_rf,
                  delete_col_rf, use_best_param, state, patient_predict, period_rf, split_size_rf, keep_psychose_rf,):
        
        # Copy the data train & test because we may modify data (drop columns) and it must not modify the orign=inal data
        # Note that the dataframes are already processed according ML parameters (keep psychose, test size, etc)
        x_train_rf, y_train_rf, x_test_rf, y_test_rf = deepcopy(self.x_train), deepcopy(self.y_train), deepcopy(self.x_test), deepcopy(self.y_test)
        
        # Drop the features the user doesn't want to keep in the data
        if delete_col_rf != []:
            if patient_predict:
                self.df_patient_pred = self.df_patient_pred.drop(delete_col_rf, axis=1)
            x_train_rf, x_test_rf = delete_col_decision_tree(delete_col_rf, x_train_rf, x_test_rf)

        # Whether the parameters of the random forest are the best from the grid search or whether the user choose the param himself
        if use_best_param==False:  #if the user choose the parameters of the random forest
            random_forest = creation_random_forest(x_train_rf, y_train_rf, n_estimators, max_depth_rf,
                                                min_samples_split_rf, min_samples_leaf_rf, max_features_rf, False) 
            
        elif use_best_param==True: #if the user choose to use the best parameters for the random forest from the grid search
            # TODO logical issue
            if delete_col_rf != []:
                grid_score_rf = random_forest_grid_search(x_train_rf, y_train_rf, 'accuracy')
                best_random_forest = grid_score_rf.best_estimator_
                
            random_forest = best_random_forest

        list_feature_importances, confu_matrix_rf, classi_report_rf, multilab_confu_mat_rf = evaluate_random_forest(x_train_rf, y_train_rf, x_test_rf, y_test_rf, random_forest)
        
        # Format the evaluation matrices
        L=list_feature_importances

        df_feature_importances_ = pd.DataFrame({'modalités':[str(L[i][1][0])+': '+str(L[i][1][1]) for i in range(len(L))], 'importance':[str(round(L[i][0]*100,1))+'%' for i in range(len(L))]})
        
        classi_report_rf = str(classi_report_rf).split('\n')
        
        if keep_psychose_rf==True :
            confu_matrix_rf = processing_matrix_format(confu_matrix_rf, self.classes_name_with_psy, confu_matrix=True, multilabel_confu_m=False)
            classi_rep_rf_1, classi_rep_rf_2, classi_rep_rf_3, classi_rep_rf_4, classi_rep_rf_5 = classi_report_rf[0],classi_report_rf[1], classi_report_rf[2], classi_report_rf[3], classi_report_rf[4] 
            classi_rep_rf_6, classi_rep_rf_7, classi_rep_rf_8, classi_rep_rf_9, classi_rep_rf_10 = classi_report_rf[5], classi_report_rf[6], classi_report_rf[7], classi_report_rf[8], classi_report_rf[9]
            multilab_confu_mat_rf = processing_matrix_format(multilab_confu_mat_rf, self.classes_name_with_psy, confu_matrix=False, multilabel_confu_m=True)
        
        elif keep_psychose_rf==False:
            confu_matrix_rf = processing_matrix_format(confu_matrix_rf, self.classes_name_without_psy, confu_matrix=True, multilabel_confu_m=False)
            classi_rep_rf_1, classi_rep_rf_2, classi_rep_rf_3, classi_rep_rf_4, classi_rep_rf_5 = classi_report_rf[0],classi_report_rf[1], classi_report_rf[2], classi_report_rf[3], classi_report_rf[4] 
            classi_rep_rf_6, classi_rep_rf_7, classi_rep_rf_8, classi_rep_rf_9, classi_rep_rf_10 = classi_report_rf[5], classi_report_rf[6], classi_report_rf[7], '', ''
            multilab_confu_mat_rf = processing_matrix_format(multilab_confu_mat_rf, self.classes_name_without_psy, confu_matrix=False, multilabel_confu_m=True)
        
        # prediction of the patients step aside
        prediction_rf = 'Il n\'y a pas eu de prédiction réalisée'

        if patient_predict:
            patient_predict = int(patient_predict)
            prediction_pat = random_forest.predict(self.df_patient_pred)
            print("prediction", prediction_pat)
            try:
                if prediction_pat[1]==1:
                    prediction_rf = "D'après le modèle, le patient {} appartient à la classe \"pas de risque\"".format(str(patient_predict))
                elif prediction_pat[1]==3:
                    prediction_rf = "D'après le modèle, le patient {} appartient à la classe \"à risque\"".format(str(patient_predict))
                elif prediction_pat[1]==5:
                    prediction_rf = "D'après le modèle, le patient {} appartient à la classe \"psychose\"".format(str(patient_predict))
            except:
                print("erreur dans prediction")
            
        if keep_psychose_rf==True :
            return(generate_table(df_feature_importances_), confu_matrix_rf[0], confu_matrix_rf[1], confu_matrix_rf[2],
                classi_rep_rf_1, classi_rep_rf_2, classi_rep_rf_3, classi_rep_rf_4, classi_rep_rf_5,
                classi_rep_rf_6, classi_rep_rf_7, classi_rep_rf_8, classi_rep_rf_9, classi_rep_rf_10,
                multilab_confu_mat_rf[0],multilab_confu_mat_rf[1],multilab_confu_mat_rf[2],multilab_confu_mat_rf[3],
                multilab_confu_mat_rf[4],multilab_confu_mat_rf[5],multilab_confu_mat_rf[6],multilab_confu_mat_rf[7], prediction_rf)

        elif keep_psychose_rf==False:
            return(generate_table(df_feature_importances_), confu_matrix_rf[0], confu_matrix_rf[1], '',
                classi_rep_rf_1, classi_rep_rf_2, classi_rep_rf_3, classi_rep_rf_4, classi_rep_rf_5,
                classi_rep_rf_6, classi_rep_rf_7, classi_rep_rf_8, classi_rep_rf_9, classi_rep_rf_10,
                multilab_confu_mat_rf[0],multilab_confu_mat_rf[1],multilab_confu_mat_rf[2],multilab_confu_mat_rf[3],
                multilab_confu_mat_rf[4],'','','', prediction_rf)


    def process_pipelines(self, *args, **kwargs):
        self.df_data, self.df_label = pipeline_preprocessing('raw_data.csv', option_patients_lost=2, period=self.period)
        print("DF DATA", self.df_data.iloc[10,10])
        self.df_color_all_lab = apply_color_label(self.df_label)
        self.df_data_disj = pipeline_disjunctive_df_data(self.df_data)
        self.df_label_disj = pipeline_disjunctive_df_label(self.df_label)
        # Init the dataframes used for the ML tasks
        self.df_data_disj_ml = deepcopy(self.df_data_disj)
        self.df_label_ml = deepcopy(self.df_label)
        self.list_df_data_disj = [self.df_data_disj]*5  # Need to be modify if we really want to take the time evolution into account
        (
            self.table_modalities_mca,
            self.table_modalities_mca_contribution,
            self.table_patients_mca,
            self.table_explained_mca,
            self.list_table_patients_mca_time
         ) = pipeline_mca(self.list_df_data_disj, 0, self.df_label_disj, nb_factors=10, benzecri=False)
         # Init the train and test data but it is overwritten
        self.x_train, self.y_train, self.x_test, self.y_test = split_train_test(self.df_data_disj_ml, self.df_label_ml, self.period, self.test_size, keep_psychose=True)

    def display_list_chi2_correlation(self, threshold_chi2, minimum_elements, chi2_label):
        print("CALLBACK CHI2")
        # Process and preprare the dataframes
        df_label_split = modify_df_label_chi2(self.df_label)
        df_test_chi2 = pd.concat([self.df_data, df_label_split], axis=1)
        # Apply chi2 test
        df_chi2, df_p_chi2 = chi2_table(df_test_chi2)
        print("CHI2 done", df_chi2.head())
        threshold_float = float(threshold_chi2)
        min_el = int(minimum_elements)
        list_corre = correlation_revealed(df_test_chi2, threshold_float,  min_el, df_p_chi2, chi2_label)
        str_list_corre_1 = str(list_corre[:len(list_corre)//3]).replace('), (',')\n(')
        str_list_corre_2 = str(list_corre[len(list_corre)//3:2*len(list_corre)//3]).replace('), (',')\n(')
        str_list_corre_3 = str(list_corre[2*len(list_corre)//3:]).replace('), (',')\n(')
        return (str_list_corre_1, str_list_corre_2, str_list_corre_3)

# Utils functions

def transform_dict_weight(dic_as_str):
    """
    Transform a string of a dictionary into a dictionary and change its keys to be integer
    """
    dic = ast.literal_eval(dic_as_str)
    new_dic={}
    for k in dic.keys():
        new_dic[int(k)]=dic[k]  # change string to int
    return new_dic


def processing_matrix_format(element, classes_name, confu_matrix=False, multilabel_confu_m=False):
    """ This function takes in argument an element (typically an array, for example a confusion matrix)
    and will split it in i new elements (i=nb of line) stored in a list 'element_str'
    
    Option 'confu_matrix' : add the class of the sample at the left of the matrix
    """
    element_str = str(element).split('\n')

    if confu_matrix == True:
        for i,row in enumerate (element_str):
            element_str[i] = classes_name[i]+'  '+row 
            
    
    if multilabel_confu_m==True:
        i=0
        for classe in classes_name:
            element_str[i] = 'Pas dans la classe : '+classe+'  '+element_str[i] 
            element_str[i+1] = 'Appartient à la classe : '+classe+'  '+element_str[i+1]
            i = i+len(classes_name) 
    
    return element_str

def generate_table(dataframe, max_rows=130):
    return html.Table(
      # Header
      [html.Tr([html.Th(col) for col in dataframe.columns])] +
      # Body
      [html.Tr([
         html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
      ]) for i in range(min(len(dataframe), max_rows))]
   )

def convert_tring_to_tuple(string_text):
    """
    String_text a string coming from str(tuple_obj)
    """
    return string_text.replace('(','').replace(')','').replace('\'','').replace(', ',',').split(',')

