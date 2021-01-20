import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from dash.dependencies import Input, Output

from preprocessing import pipeline_preprocessing
from disjunctive_array.pipeline import pipeline_disjunctive_df_data, pipeline_disjunctive_df_label
from mca.pipeline import pipeline_mca
from ml.utils import split_train_test
from statistics.chi2 import correlation_revealed, modify_df_label_chi2, chi2_table

import disjunctive_array
import mca
import preprocessing
import ml
import statistics
import visualisation

class GenerateApp():
    """
    """
    html = []
    classes_name_with_psy = ["pas de risque","a risque","psychose"]
    classes_name_without_psy = ["pas de risque","a risque"]
    list_moda = [('label', 'psychose')]  #default list_modalities
    period = 4
    test_size = 0.2
    x_train, y_train, x_test, y_test = pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(),  pd.DataFrame()
    df_data, df_label, df_data_disj = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    df_data_disj_patients_pred = pd.DataFrame() # pd.DataFrame(0, index=[0], columns=df_data_disj.columns)
    df_data_disj_rf = pd.DataFrame() # deepcopy(df_data_disj)
    df_label_rf = pd.DataFrame() # deepcopy(df_label)

    def __init__(self, *args, **kwargs):
        print("INIT KWARGS", kwargs)
        self.app = kwargs.get('app')

        self.process_pipelines()

        self.html = [
            html.H1(className="app-title", children='Prévention en santé mentale'),
            html.Div(id="all-the-data",
                children=[ 
                    html.Div(id='param-global-var',
                        children=[
                            html.H4(
                                 className="margin-bot-8", 
                                children="Paramètres de l'application"
                            ),
                            html.P(children="Veuillez choisir le nom du fichier csv dans le répertoir courant"),
                            dcc.Input(
                                id='name_file',
                                value='raw_data.csv',
                                type='text'
                            ),
                            html.P(children="Voulez-vous conserver les patients perdus de vue au cours de l'étude \
                                (après avoir au moins effectués la première consultation) en les \
                                considérant avec le même diagnostic qu'à leur dernière consultation ou les enlever de l'étude ?"
                            ),
                            dcc.RadioItems(
                                id='option_lost',
                                options=[
                                    {'label':'garder les patients perdu de vue', 'value':2},
                                    {'label':'retirer les patients perdus de vue ', 'value':3}],
                                value=2,
                                labelStyle={'display': 'inline-block'}
                            ),
                            html.P(
                                className="hidden-div",
                                id='hidden_div'
                            ),
                        ]
                    ),
                    html.Div(id="test chi2",
                        children=[
                            html.H2(className="h2-chi2", children="Test du chi2"),
                            html.H6(
                                className="h6-chi2",
                                children="Le test du chi2 permet de savoir si deux variables qualitatives sont indépendantes ou non, c'est à dire si les réponses de l'une dépendent de l'autre.\
                                        On formule l'hypothèse H0 : il y a indépendance entre les 2 variables. On fixe un seuil de confiance, habituellement 0.05.\
                                        Les variables qui sont affichées ci-dessous ont toutes une hypothèse H0 rejetée. Elles ne sont donc pas indépendantes.",
                            ),
                            html.H6(children="Veuillez choisir un seuil de confiance et le nombre minimal d'éléments \
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
                            html.H6(children="Voulez-vous faire le test du khi2 uniquement entre les variables et le label ?"),
                            dcc.RadioItems(
                                id="chi2_label",
                                options=[
                                    {'label':'oui', 'value':True},
                                    {'label':'non', 'value':False}],
                                value=False,
                                labelStyle={'display': 'inline-block'}
                            ),
                            html.H6(
                                id="explication-chi2", 
                                className="margin-20", 
                                children="Voici ci dessous la liste des variables qui sont \
                                    associées entre elles (2 par 2). On ne peut cependant pas savoir dans quel sens varie cette dépendance."
                            ),
                            html.Div(
                                id='resultat chi2', 
                                style={"display":"grid", "grid-template-columns": "auto auto auto", 'column-fill':'auto',"grid-gap": "8px",
                                        "grid-column-gap": "3%", "white-space":"pre"},
                                children=[
                                    html.P(id="list of correlation1"),
                                    html.P(id="list of correlation2"),
                                    html.P(id="list of correlation3")
                                ]
                            )           
                        ],
                        style={"margin-bottom":"25px"}
                    ),
                #     html.Div(id="all the ACM",
                #         children=[
                #             html.H2(
                #                 children="Analyse des correspondances multiples",
                #                 style={"font-family": "Open Sans", "margin-left":"35%"}
                #             ),
                #             html.P(children="Effectif d'une modalité "),
                #             dcc.Dropdown(
                #                 id="effective_modality",
                #                 options=[{'label':str(i) ,'value':i} for i in self.df_data_disj.columns],
                #                 placeholder="Choisissez une modalité"     
                #             ),
                #             html.P(id='effective_modality_display'),

                #             html.Div(id="1er graphe ACM",
                #                 children=[
                #                     html.H3(children="Graphe des coordonnées des modalités sur le plan de facteurs de l'ACM ",
                #                         style={"align-items":"center"}
                #                     ),

                #                     html.Div(id = 'graph_var_by_var',
                #                         children=[
                #                             html.H6(children='Quel référentiel de temps voulez vous choisir pour le label ?'),
                #                                 dcc.RadioItems(
                #                                     id='choose_period_graph',
                #                                     options=[
                #                                         {'label':'M0', 'value':0},
                #                                         {'label':'M6', 'value':1},
                #                                         {'label':'M12', 'value':2},
                #                                         {'label':'M18', 'value':3},
                #                                         {'label':'M24', 'value':4}
                #                                     ],
                #                                     value=0,
                #                                     labelStyle={'display' : 'inline-block'}
                #                                 ),

                #                                 html.H6(children="Choisissez le facteur pour l'axe des abscisses"),

                #                                 dcc.RadioItems(
                #                                     id='fs_id1_',
                #                                     options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                     value=1,
                #                                     labelStyle={'display': 'inline-block'}
                #                                 ),

                #                                 html.H6(children="Choisissez le facteur pour l'axe des ordonnées"),

                #                                 dcc.RadioItems(
                #                                     id='fs_id2_',
                #                                     options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                     value=2,
                #                                     labelStyle={'display': 'inline-block'}
                #                                 ),
                                                
                #                                 html.H6(children="Voulez-vous afficher uniquement les modalités significatives ?"),
                #                                 dcc.RadioItems(
                #                                     id='significant_only',
                #                                     options=[
                #                                         {'label': "oui", 'value': True},
                #                                         {'label':"non", 'value':False}
                #                                     ],
                #                                     value=False,
                #                                     labelStyle={'display': 'inline-block'}
                #                                 ),
                #                                 html.Button(
                #                                     id='submit-button-state_3', n_clicks=0, children='Afficher le graphe',
                #                                     style={'margin-top':'12px'}
                #                                 ),
                #                                 dcc.Loading(
                #                                     id="loading-3",
                #                                     type="default",
                #                                     children= dcc.Graph(
                #                                         id='var_by_var', 
                #                                         config ={'autosizable' :True, 'responsive':False},
                #                                         style={"postion":"relative"}
                #                                     )
                #                                 )                                             
                #                             ]

                #                     )
                #                 ]
                #             ),
                            
                #             html.Div(
                #                 id="explanation graphe modalities",
                #                 children=[
                #                     html.H4(children="Analyse des coordonées des modalités du graphe"),
                #                     html.Div(id="hidden_div4", style={"display":"none"}),
                #                     html.P(children="Veuillez choisir une modalité dont on cherchera les modalités proches dans le graphe"),

                #                     dcc.Dropdown(
                #                         id="modality_to_evaluate",
                #                         value=('label','psychose')  #value by default
                #                     ),
                #                     html.P(children="Veuillez choisir la distance euclidienne maximum avec les autres modalités"),

                #                     dcc.Slider(
                #                         id='dist_moda',
                #                         min=0.0,
                #                         max=20,
                #                         marks={i: '{}'.format(round(i/10,1)) for i in range(0,21)},
                #                         value= 6
                #                     ),
                                    
                #                     html.H4(id="text_before_array"),
                #                     dcc.Loading(
                #                         id="loading-4",
                #                         type="default",
                #                         children= html.P(
                #                             id='array_graph_moda', 
                #                             style={'margin-top':'20px','margin-bottom':'20px','margin-left':'420px'}
                #                         )
                #                     )
                #                 ]
                #             ),
                #             html.Div(
                #                 id='analysis factors',
                #                 style ={'margin-bottom':'20px'},
                #                 children=[
                #                     html.H3(children="Analyse d'un facteur"),
                #                     html.P(children='Choisissez le facteur que vous vouler étudier'),
                #                     dcc.RadioItems(
                #                         id='factor_to_analyse',
                #                         options=[{'label': i, 'value': i} for i in range(1,11)],
                #                         value=1,
                #                         labelStyle={'display': 'inline-block'}
                #                     ),
                #                     html.Div(
                #                         id="arrays analys factor",
                #                         style={"display":"grid", "grid-template-columns": "auto auto", "grid-gap": "8px", "grid-column-gap": "3%", "white-space":"pre"},
                #                         children=[
                #                             html.P(
                #                                 id='modalities_coordo_negative_1', 
                #                                 style={'margin-left':'auto', 'margin-right':'auto'}
                #                             ),
                #                             html.P(
                #                                 id='modalities_coordo_positive_1',
                #                                 style={'margin-left':'auto', 'margin-right':'auto'}
                #                             )     
                #                         ]
                #                     ),
                #                     html.P(
                #                         id='modalities_contribution_1', 
                #                         style={'margin-top':'20px','margin-bottom':'20px', 'margin-left':'420px'}
                #                     ),
                #                     html.P(
                #                         id='sentence_negative',
                #                         style={'margin-bottom':'18px'}
                #                     ),
                #                     html.P(
                #                         id='sentence_positive',
                #                         style={'margin-bottom':'18px'}
                #                     ),
                #                     html.Div(
                #                         id='graph_patient_feature',
                #                         children=[
                #                             html.H3(
                #                                 children="Graphe des coordonnées des patients dans le plan des facteurs de l'ACM selon la variable"
                #                             ),
                #                             html.P(children='Choisissez le facteur pour l\'axe des abscisses'),
                #                             dcc.RadioItems(
                #                                 id='factor_abs_graphe_pat_var',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=1,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.P(children='Choisissez le facteur pour l\'axe des ordonnées'),
                #                             dcc.RadioItems(
                #                                 id='factor_ordo_graphe_pat_var',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=2,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.P(children="Choisissez une variable"),
                #                             dcc.Dropdown(
                #                                 id="choose_feature",
                #                                 options=[{'label':str(i) ,'value':str(i)} for i in df_data.columns],
                #                                 value='dependance'
                #                             ),
                #                             dcc.Graph(id='patients_var', config ={'autosizable' :True, 'responsive':False})
                                                    
                #                         ]
                #                     )
                                        
                #                 ]
                #             ),

                #             html.Div(
                #                 id="2e graphe ACM",
                #                 children=[
                #                     html.H3(children="Graphe des coordonnées des patients sur le plan des facteurs de l'ACM "),
                #                     html.Div(
                #                         id = 'graph_patients',
                #                         children=[
                #                             html.H6(children="Choisissez le facteur pour l'axe des abscisses"),
                #                             dcc.RadioItems(
                #                                 id='fs_id1_p',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=1,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.H6(children="Choisissez le facteur pour l'axe des ordonnées"),
                #                             dcc.RadioItems(
                #                                 id='fs_id2_p',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=2,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.H6(children="Classe des patients"),
                #                             dcc.Checklist(
                #                                 id='class_patient_2D',
                #                                 options=[
                #                                     {'label':'pas de risque', 'value':1},
                #                                     {'label':'a risque', 'value':3},
                #                                     {'label':'psychose', 'value':5}
                #                                 ],
                #                                 value=[1,3,5],
                #                                 labelStyle={'display':'inline-block'}
                #                             ),
                #                             html.H6(children="Choississez les patients que vous voulez afficher"),
                #                             dcc.Dropdown(
                #                                 id='choose_pat_2d',
                #                                 options=[{'label':i, 'value':i} for i in range (table_patients_mca.shape[1])],
                #                                 value=[i for i in range (table_patients_mca.shape[1])],
                #                                 multi=True
                #                             ),
                #                             html.Button(
                #                                 id='submit-button-state_1', 
                #                                 n_clicks=0, 
                #                                 children='Afficher le graphe',
                #                                 style={'margin-top':'12px'}
                #                             ),
                #                             html.Hr(),
                #                             dcc.Loading(
                #                                 id="loading-1",
                #                                 type="default",
                #                                 children=dcc.Graph(id='patients_2d', config ={'autosizable' :True, 'responsive':False})
                #                             )
                #                         ]

                #                     )
                #                 ]
                #             ),
                            
                #             html.Div(
                #                 id="3e graphe ACM : patients 3D",
                #                 children=[
                #                     html.H3(children="Graphe des coordonnées des patients sur le plan en 3D des facteur de l'ACM"),
                #                     html.Div(
                #                         id='graph3D_patients',
                #                         children=[
                #                             html.H6(children="Choisissez le facteur pour l'axe des abscisses"),
                #                             dcc.RadioItems(
                #                                 id='fs_id1_p_3D',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=1,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.H6(children="Choisissez le facteur pour l'axe des ordonnées'"),
                #                             dcc.RadioItems(
                #                                 id='fs_id2_p_3D',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=2,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.H6(children="Choisissez le facteur pour l'axe z"),
                #                             dcc.RadioItems(
                #                                 id='fs_id3_p_3D',
                #                                 options=[{'label': i, 'value': i} for i in range(1,11)],
                #                                 value=3,
                #                                 labelStyle={'display': 'inline-block'}
                #                             ),
                #                             html.H6(children="Classe des patients"),
                #                             dcc.Checklist(
                #                                 id='class_patient_3D',
                #                                 options=[
                #                                     {'label':'pas de risque', 'value':1},
                #                                     {'label':'a risque', 'value':3},
                #                                     {'label':'psychose', 'value':5}],
                #                                 value=[1,3,5],
                #                                 labelStyle={'display':'inline-block'}
                #                             ),
                #                             dcc.Dropdown(
                #                                 id='list_patients_keep_3d',
                #                                 options=[{'label':i, 'value':i} for i in range (table_patients_mca.shape[1])],
                #                                 value=[i for i in range (table_patients_mca.shape[1])],
                #                                 multi=True
                #                             ),
                #                             html.Button(
                #                                 id='submit-button-state_2', 
                #                                 n_clicks=0, 
                #                                 children='Afficher le graphe',
                #                                 style={'margin-top':'12px'}
                #                             ),
                #                             html.Hr(),
                #                             dcc.Loading(
                #                                 id="loading-2",
                #                                 type="default",
                #                                 children=dcc.Graph(id='patients_3D', config ={'autosizable' :True, 'responsive':False})
                #                             ),
                #                             html.P(children="Si le patients a une coordonnée positive selon le facteur en abscisse, il aura tendance à avoir : "),
                #                             html.P(id='contributions_fs_x'),
                #                             html.P(children="Si le patients a une coordonnée négative selon le facteur en abscisse, il aura tendance à avoir : "),
                #                             html.P(id='contributions_fs_x_')    
                #                         ]

                #                     )
                #                 ]
                #             )
                #         ],
                #         style={"border-top": "2px solid black", "margin-top":"10px"}
                #     ),
                # # todo refactor 
                # html.Div(id="all of the decision tree",
                #     children=[

                #         html.Div(id="all the parameters of the tree",
                #             children=[

                #             html.Div(children=[

                #                         html.H2(children="Arbre de décision",
                #                                 style={"font-family": "Open Sans", "margin-left":"38%"}
                #                         ),
                #                         html.Div(id="specifité de l'arbre de décision",
                #                                 children=[
                #                                     html.P(children='Quel référentiel de temps voulez vous choisir pour le diagnostic du patient ?'
                #                                     ),
                #                                     dcc.RadioItems(id='choose_period_decision_tree',
                #                                                 options=[{'label':'M0', 'value':0},
                #                                                         {'label':'M6', 'value':1},
                #                                                         {'label':'M12', 'value':2},
                #                                                         {'label':'M18', 'value':3},
                #                                                         {'label':'M24', 'value':4}],
                #                                                 value=4,
                #                                                 labelStyle={'display' : 'inline-block'}
                #                                     ),

                #                                     html.P(children="Choississez le pourcentage des données utilisées pour tester l'arbre de décision. \
                #                                                     Le reste des données sera utilisé pour entrainer l'arbre "
                #                                     ),
                #                                     dcc.RadioItems(id='split_size',
                #                                                 options=[{'label':'10%', 'value':0.1},
                #                                                         {'label':'20%', 'value':0.2},
                #                                                         {'label':'30%', 'value':0.3},
                #                                                         {'label':'40%', 'value':0.4}],
                #                                                 value=0.3,
                #                                                 labelStyle={'display' : 'inline-block'}
                #                                     ),
                #                                     html.P(children="Voulez-vous garder les patients atteints de psychose de l'étude ?"
                #                                     ),
                #                                     dcc.RadioItems(id='keep_psychose',
                #                                                 options=[{'label':'oui', 'value':True},
                #                                                         {'label':'non', 'value':False}],
                #                                                 value=True,
                #                                                 labelStyle={'display': 'inline-block'}
                #                                     ),

                #                                     html.P(id='hidden_div2', style={'display':'none'}
                #                                     ),
                #                                 ]
                #                         ),
                                            
                #                         html.Div(id="div for the parameters of the tree",
                #                                 children=[
                #                                     html.Div(style = {"display":"grid", "grid-template-columns": "auto auto auto auto auto auto",
                #                                                     "grid-gap": "8px",
                #                                                     "grid-column-gap": "3%",
                #                                                     "white-space":"pre"},
                #                                             children=[
                #                                                 html.P(children='Les paramètres de l\'arbre permettant d\'avoir le meilleur score de la mesure suivante sont :'
                #                                                 ),
                #                                                 dcc.RadioItems(id="scoring_best_para",
                #                                                             options=[{'label': 'accuracy', 'value': 'accuracy'},
                #                                                             {'label':'f1_macro', 'value': 'f1_macro'}],
                #                                                         value ="accuracy",
                #                                                         labelStyle={'display' : 'inline-block'}
                #                                                 )
                #                                             ]
                #                                     ),
                #                                     html.P(id='best_para_tree'
                #                                     ),
                #                                     html.H5(children="Vous pouvez régler les paramètres de l'arbre ci-dessous :",
                #                                             style={'margin-top':'25px'}
                #                                     ),

                                                    
                #                                     html.P(children="Profondeur maximale de l'arbre :",
                #                                         style={'margin-top':'8px'}
                #                                     ),
                #                                     dcc.RadioItems(
                #                                         id='depth_',
                #                                         options=[{'label':i, 'value': i} for i in [1,2,3,4,5]],
                #                                         value=3,
                #                                         labelStyle={"display":"inline-block"}
                #                                     ),
                #                                     html.P(children="Poids des classes :", style={'margin-top':'12px'}
                #                                     ),
                #                                     html.Div(id="div_weight_with_psychose",
                #                                             children=[
                #                                                 dcc.RadioItems(
                #                                                     id='weight_with_psychose',
                #                                                     options=[{'label': "pas de poids", 'value' : {'1': 1,  '3': 1, '5': 1}},
                #                                                             {'label': "classe 'pas de risque' accentuée", 'value' : {'1': 10,  '3': 1, '5': 1}},
                #                                                             {'label': "classe 'a risque' accentuée", 'value' : {'1': 1,  '3': 10, '5': 1}},
                #                                                             {'label': "classe 'psychose' accentuée", 'value' : {'1': 1,  '3': 1, '5': 10}},
                #                                                                 ],
                #                                                     value={'1': 1,  '3': 1, '5': 1},
                #                                                     labelStyle={"display":"inline-block"}
                #                                                 )
                #                                             ]
                #                                     ),
                #                                     html.Div(id="div_weight_without_psychose",
                #                                             children=[
                #                                                 dcc.RadioItems(
                #                                                     id='weight_without_psychose',
                #                                                     options=[{'label': "pas de poids", 'value' : {'1': 1,  '3': 1}},
                #                                                         {'label': "classe 'pas de risque' accentuée", 'value' : {'1': 10,  '3': 1}},
                #                                                         {'label': "classe 'a risque' accentuée", 'value' : {'1': 1,  '3': 10}}],
                #                                                     value={'1': 1,  '3': 1},
                #                                                     labelStyle={"display":"inline-block"}
                #                                                 )
                #                                             ]
                #                                     ),


                #                                     html.P(children="Nombre minimal de patients requis à chaque noeud :",
                #                                         style={"margin-top":"12px"}
                #                                     ),
                #                                     dcc.Slider(
                #                                         id='min_split_',
                #                                         min=2,
                #                                         max=10,
                #                                         marks={i: 'min_split {}'.format(i) for i in range(2,11)},
                #                                         value=5
                #                                     ),
                #                                     html.P(children="Nombre minimal de patients requis à chaque feuille :",
                #                                         style={"margin-top":"12px"}
                #                                     ),
                #                                     dcc.Slider(
                #                                         id='min_leaf_',
                #                                         min=1,
                #                                         max=10,
                #                                         marks={i: 'min_leaf {}'.format(i) for i in range(1,11)},
                #                                         value=2
                #                                     ),
                #                                     html.P(children="Variable(s) à ne pas prendre en compte pour l'arbre :",
                #                                         style={'margin-top':'12px'}
                #                                     ),

                #                                 dcc.Dropdown(id='delete_col',
                #                                         options=[{'label': i, 'value': i} for i in x_train.columns.get_level_values('features').unique().to_list()],
                #                                         multi=True,
                #                                         value="False"
                #                                     ) 
                #                                 ]
                #                             )


                #             ])


                #         ]),

                #         html.Img(id='decision tree', style={"margin-bottom":"20px", "margin-top":"20px"} 
                #         ),
                #         html.P(id="explication_arbre", children="L'arbre de décision essaie de séparer les différentes classes 'pas de risque', \
                #         'a risque' et 'psychose' à chaque noeud. L'objectif est de trouver des combinaisons de variables permettant d'avoir des feuilles \
                #         les plus pures possible, c'est à dire ayant uniquement des patients de la même classe. "
                #         ),
                        
                        
                #         html.Div(id='analysis prediction tree',
                #             children=[
                #                 html.H2(children="Etude de l'arbre de décision : test de prédiction "
                #                 ),
                                
                #                 html.H5(id="confusion matrix title", children="Matrice de confusion de l'arbre de décision"
                #                 ),
                                
                #                 html.P(id="explication_tree", children="Les lignes de la matrice de confusion représentent les classes\
                #                     de l'étude et la somme de chaque ligne le nombre d'éléments pour la classe. \
                #                     Les colonnes représentent les classes prédites. Idéalement, tous les éléments se retrouvent sur la \
                #                     diagonales : ils ont été prédits dans leur classe.", style={"margin-bottom":"20px"}
                #                 ),
                                
                #                 html.Div(id="confusion matrix",
                #                         style={"display":"grid", "margin-right":"70%","font-size":"20px",
                #                             "text-align-last": "right","white-space":"pre"},
                #                         children=[
                #                             html.P(id="column", children=["Prédictions"]
                #                             ),
                #                             html.P(id="confu_matrix_1"
                #                             ),
                #                             html.P(id="confu_matrix_2"
                #                             ),
                #                             html.P(id="confu_matrix_3"
                #                             )
                #                         ]),
                #                 html.Div(id="classi_report",
                #                         children=[
                #                             html.H5(id="classifiaction report title", children="Analyse des résultats de la prédiction"
                #                             ),
                #                             html.P(id="explication_classi", children=" explication"
                #                             ),
                #                             html.Div(id="classification report",
                #                                     style={'display':'grid',"margin-right":"70%","font-size":"15px",
                #                                         "text-align-last": "right","white-space":"pre"},
                #                                     children=[
                #                                         html.P(id="classi_rep_1"),
                #                                         html.P(id="classi_rep_2"),
                #                                         html.P(id="classi_rep_3"),
                #                                         html.P(id="classi_rep_4"),
                #                                         html.P(id="classi_rep_5"),
                #                                         html.P(id="classi_rep_6"),
                #                                         html.P(id="classi_rep_7"),
                #                                         html.P(id="classi_rep_8"),
                #                                         html.P(id="classi_rep_9"),
                #                                         html.P(id="classi_rep_10")
                #                                     ]),
                #                         ]),
                #                 html.Div(id="multilabel_confusion_matrix",
                #                         children=[
                #                             html.H5(id="mutlilabel title", children="Matrice de confusion pour chaque classe"
                #                             ),
                #                             html.P(id="explication_multilab", children=" La matrice de confusion est calculée pour chaque \
                #                             classe."
                #                             ),
                #                             html.Div(id="mutlilabel_confu_mat",
                #                                     style={'display':'grid',"margin-right":"70%","font-size":"15px",
                #                                         "text-align-last": "right","white-space":"pre"},
                #                                     children=[
                #                                         html.P(id="multi_confu_1"),
                #                                         html.P(id="multi_confu_2"),
                #                                         html.P(id="multi_confu_3"),
                #                                         html.P(id="multi_confu_4"),
                #                                         html.P(id="multi_confu_5"),
                #                                         html.P(id="multi_confu_6"),
                #                                         html.P(id="multi_confu_7"),
                #                                         html.P(id="multi_confu_8")
                                                        
                #                                     ]),
                #                         ])
                            
                                
                #             ])

                #     ],
                #     style={"margin-top":"20px","border-top": "2px solid black"}
                #     ),
                    
                #     html.Div(id="Random forest", style={"border-top": "2px solid black"},
                #             children=[
                #                 html.H2(children="Random forest", style={"font-family": "Open Sans", "margin-left":"38%",
                #                                                         "margin-top":"20px"}
                #                 ),
                                
                #                 html.P(id="explication_rf_all", children=" La forêt aléatoire est une technique d'apprentissage supervisé \
                #                 basée sur l'utilisation de nombreux arbres de décision. Chaque arbre prend seulement une partie des données \
                #                 pour s'entrainer, différente entre chaque arbre. L'autre partie de données est utilisée pour le tester. Ainsi, \
                #                 chaque arbre sera entrainé et testé sur des données différentes. Par ailleurs, chaque arbre a un nombre limité \
                #                 de modalités disponibles pour faire les tests de séparation. Il choisira a chaque noeud la meilleure parmi celles à \
                #                 disposition. Cela permet de forcer l'utilisation de différentes modalités et de tester différents enchainement de tests. \
                #                 "),
                #                 html.Div(id="prediction_rf", style={'margin-top':'20px', 'margin-bot':'20px'},
                #                         children=[
                #                             html.P('Voulez-vous mettre un patient de côté pour faire une prédiction de sa classe ?'
                #                             ),
                                            
                #                             dcc.RadioItems(id="keep_patient_aside",
                #                                         options=[{'label':'oui', 'value':True},
                #                                                 {'label':'non', 'value':False}],
                #                                         value = True,
                #                                         labelStyle={"display":"inline-block"}
                #                             ),
                #                             html.Div(id="display_patient_to_choose",
                #                                     children=[
                #                                         html.P("Choisissez le patient que vous voulez mettre de côté pour ensuite prédire sa classe"
                #                                         ),
                #                                         dcc.Dropdown(id='patients_to_evaluate',
                #                                                 options=[{'label':i, 'value':i} for i in (df_data_disj.index.to_list())],
                #                                                 value=None
                                                                
                #                                         ), 
                #                                         html.Div(id="hidden_div5", style={'display':'none'}
                #                                         )
                                                        
                                                        
                #                                     ]
                #                             )
                                            
                                            
                #                             ]
                #                 ),
                                
                #                 html.P("Choisissez la période d'évaluation :"
                #                 ),
                #                 dcc.RadioItems(id="period_rf",
                #                             options=[{'label':'M0', 'value':0},
                #                                     {'label':'M6', 'value':1},
                #                                     {'label':'M12', 'value':2},
                #                                     {'label':'M18', 'value':3},
                #                                     {'label':'M24', 'value':4}],
                #                             value=4,
                #                             labelStyle={"display":"inline-block"}
                #                 ),
                                
                #                 html.H6(children="Voulez-vous garder les patients diagnostiqués en \"psychose\" de l'étude ?"
                #                 ),
                                
                #                 dcc.RadioItems(id="keep_psychose_rf",
                #                             options=[ {'label':'oui', 'value':True},
                #                                         {'label':'non', 'value':False}],
                #                             value=True,
                #                             labelStyle={'display': 'inline-block'}
                #                 ),
                                
                #                 html.H6(children="Pourcentage des données utilisées pour la partie test"
                #                 ),
                                
                #                 dcc.RadioItems(id="split_size_rf",
                #                             options=[{'label':'10%', 'value':0.1},
                #                                         {'label':'20%', 'value':0.2},
                #                                         {'label':'30%', 'value':0.3}],
                #                             value=0.2,
                #                             labelStyle={"display":"inline-block"}
                #                 ),
                #                 html.Div(id="hidden_div3", style={"display":"none"}
                #                 ),
                                
                #                 html.Div(id="best_param_random_forest", 
                #                         children=[
                #                         html.H4("Calcul des meilleurs paramètres"
                #                         ),
                                            
                #                         html.P("Sur quel scoring voulez-vous calculer ces paramètres (accuracy, f1_score):"
                #                         ),
                                            
                #                         dcc.Input(id='scoring_best_param_random_forest', value='accuracy', type='text'
                #                         ),
                #                         html.P("Voici le meilleur paramétrage :"
                #                         ),    
                #                         html.P(id = "best_param_random_forest_display"
                #                         )
                #                         ]
                #                 ),
                                

                #                 html.Div(id="all the parameters of the rf",
                #                     children=[
                                        
                #                         html.P("Voulez-vous choisir les meilleurs paramètres pour la forêt aléatoire ? " 
                #                             ),  
                #                             dcc.RadioItems(
                #                                 id="use_best_param",
                #                                 options=[{'label':'non', 'value': False},
                #                                         {'label':'oui', 'value': True}],
                #                                 value=False,
                #                                 labelStyle={"display":"inline-block"}
                #                             ),

                #                         html.Div(children=[
                #                             html.P("Profondeur maximale de l'arbre :"
                #                             ),
                #                             dcc.RadioItems(
                #                                 id='max_depth_rf',
                #                                 options=[{'label':i, 'value': i} for i in [2,3,4,'none']],
                #                                 value=3,
                #                                 labelStyle={"display":"inline-block"}
                #                             ),
                                            
                #                             html.P("Nombre d'arbres de décision dans la forêt aléatoire :"
                #                             ),
                #                             dcc.RadioItems(
                #                                 id='n_estimators',
                #                                 options=[{'label':10, 'value': 10},
                #                                         {'label':20, 'value': 20},
                #                                         {'label':50, 'value': 50},
                #                                         {'label':100, 'value': 100}],
                #                                 value=50,
                #                                 labelStyle={"display":"inline-block"}
                #                             ),
                #                             html.P("Nombre maximal de modalités disponibles pour un arbre :"
                #                             ),
                                            
                #                             dcc.RadioItems(
                #                                 id='max_features_rf',
                #                                 options=[{'label':10, 'value': 10},
                #                                         {'label':30, 'value': 30},
                #                                         {'label':50, 'value': 50},
                #                                         {'label':70, 'value': 70},
                #                                         {'label':100, 'value': 100}],
                #                                 value=30,
                #                                 labelStyle={"display":"inline-block"}
                #                             ),
                                        
                #                             html.P("Nombre minimum de patients en sortie de noeud "
                #                             ),
                #                             dcc.Slider(
                #                                 id='min_samples_split_rf',
                #                                 min=2,
                #                                 max=10,
                #                                 marks={i: 'min_split {}'.format(i) for i in range(2,11)},
                #                                 value=5
                #                             ),
                                            
                #                             html.P("Nombre minimum de patients dans une feuille "
                #                             ), 
                                            
                #                             dcc.Slider(
                #                                 id='min_samples_leaf_rf',
                #                                 min=1,
                #                                 max=10,
                #                                 marks={i: 'min_leaf {}'.format(i) for i in range(1,11)},
                #                                 value=5
                #                             ),
                                            
                                        
                #                             html.P("Voulez-vous retirer des variables du modèle ? "
                #                             ), 
                #                             dcc.Dropdown(id='delete_col_rf',
                #                                 options=[{'label': i, 'value': i} for i in x_train_rf.columns.get_level_values('features').unique().to_list()],
                #                                 multi=True,
                #                                 value=[]
                #                             ) 
                #                 ])           
                #             ]),
                                
                #             html.Div(id='analysis random forest',
                #                     children=[
                #                         html.H2(children="Etude de la forêt aléatoire : test de prédiction ", style={'margin-bottom':"20px"}
                #                         ),
                                        
                #                         html.H5("Résultats de la prédiction :"
                #                         ),
                #                         dcc.Loading(id="loading-6",
                #                                     type="default",
                #                                     children=  html.P(id="results_prediction_rf", style={"margin-bottom":"15px"})
                                        
                #                         ),
                                        
                                        
                #                         html.H5(id='list_feature_importances_title', children="Importance des variables dans la forêt aléatoire "
                #                         ),
                #                         html.P(id="list_feature_importances"
                #                         ),
                                
                #                         html.H5(id="confusion matrix rf title", children="Matrice de confusion de la forêt aléatoire"
                #                         ),
                                
                #                         html.P(id="explication_rf", children="Les lignes de la matrice de confusion représentent les classes\
                #                             de l'étude et la somme de chaque ligne le nombre d'éléments pour la classe. \
                #                             Les colonnes représentent les classes prédites. Idéalement, tous les éléments se retrouvent sur la \
                #                             diagonales : ils ont été prédits dans leur classe.", style={"margin-bottom":"20px"}
                #                         ),
                                
                #                         html.Div(id="confusion matrix rf",
                #                                 style={"display":"grid", "margin-right":"70%","font-size":"20px",
                #                                     "text-align-last": "right","white-space":"pre"},
                #                                 children=[
                #                                     html.P(id="column_rf", children=["Prédictions"]
                #                                     ),
                #                                     html.P(id="confu_matrix_rf_1"
                #                                     ),
                #                                     html.P(id="confu_matrix_rf_2"
                #                                     ),
                #                                     html.P(id="confu_matrix_rf_3"
                #                                     )
                #                                 ]),
                #                         html.Div(id="classi_report_rf",
                #                                 children=[
                #                                     html.H5(id="classification report title", children="Analyse de la forêt aléatoire"
                #                                     ),
                #                                     html.P(id="explication_classi_rf", children=" explication"
                #                                     ),
                #                                     html.Div(id="classification report_rf",
                #                                             style={'display':'grid',"margin-right":"70%","font-size":"15px",
                #                                                 "text-align-last": "right","white-space":"pre"},
                #                                             children=[
                #                                                 html.P(id="classi_report_rf_1"),
                #                                                 html.P(id="classi_report_rf_2"),
                #                                                 html.P(id="classi_report_rf_3"),
                #                                                 html.P(id="classi_report_rf_4"),
                #                                                 html.P(id="classi_report_rf_5"),
                #                                                 html.P(id="classi_report_rf_6"),
                #                                                 html.P(id="classi_report_rf_7"),
                #                                                 html.P(id="classi_report_rf_8"),
                #                                                 html.P(id="classi_report_rf_9"),
                #                                                 html.P(id="classi_report_rf_10")
                #                                             ]),
                #                             ]),
                #                         html.Div(id="multilabel_confusion_matrix_rf",
                #                                 children=[
                #                                     html.H5(id="mutlilabel title rf", children="Matrice de confusion pour chaque classe"
                #                                     ),
                #                                     html.P(id="explication_multilab_rf", children=" La matrice de confusion est calculée pour chaque \
                #                                     classe."
                #                                     ),
                #                                     html.Div(id="mutlilabel_confu_mat_rf",
                #                                             style={'display':'grid',"margin-right":"70%","font-size":"15px",
                #                                                 "text-align-last": "right","white-space":"pre"},
                #                                             children=[
                #                                                 html.P(id="multilab_confu_mat_rf_1"),
                #                                                 html.P(id="multilab_confu_mat_rf_2"),
                #                                                 html.P(id="multilab_confu_mat_rf_3"),
                #                                                 html.P(id="multilab_confu_mat_rf_4"),
                #                                                 html.P(id="multilab_confu_mat_rf_5"),
                #                                                 html.P(id="multilab_confu_mat_rf_6"),
                #                                                 html.P(id="multilab_confu_mat_rf_7"),
                #                                                 html.P(id="multilab_confu_mat_rf_8")

                #                                             ]),
                #                                 ])


                #                 ])

                                        
                                        
                                    
                            # ])
            ],
            style = {"position": "relative", "background-color": "white", "border-radius": "2px 2px 5px 5px",
                    "font-size": "1.5rem", "box-shadow": "rgb(240, 240, 240) 5px 5px 5px 0px",
                    "border": "thin solid rgb(240, 240, 240)", "margin-left": "auto","margin-right": "auto","color": "#302F54",
                    "padding": "8px", "width": "90%", "max-width": "none","box-sizing":"border-box"
            }
        )
                    
    ]

    def process_pipelines(self, *args, **kwargs):
        self.df_data, self.df_label = pipeline_preprocessing('raw_data.csv', option_patients_lost=2, period=self.period)
        print("DF DATA", self.df_data.iloc[10,10])
        self.df_data_disj = pipeline_disjunctive_df_data(self.df_data)
        self.x_train, self.y_train, self.x_test, self.y_test = split_train_test(self.df_data_disj, self.df_label, self.period, self.test_size, keep_psychose=True)


# Utils functions

def transform_dict_weight(dic):
    """ an intermedian function used in the app below to transform the weight of the decision tree when it s a dic"""
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

