from torch import nn
from data.utils import weighted_cross_entropy


DATASET_FEATURE_STAT_DICT = {
    'zinc': {'node': 21, 'edge': 4, 'num_class': 1},  # regression
    'mutag': {'node': 7, 'edge': 4, 'num_class': 1},  # bin classification
    'alchemy': {'node': 6, 'edge': 4, 'num_class': 12},  # regression, but 12 labels

    'proteins_full': {'node': 3, 'edge': 0, 'num_class': 1},  # bin classification
    'ptc_mr': {'node': 18, 'edge': 4, 'num_class': 1},  # bin classification
    'nci1': {'node': 37, 'edge': 0, 'num_class': 1},  # bin classification
    'nci109': {'node': 38, 'edge': 0, 'num_class': 1},  # bin classification
    'imdb-multi': {'node': 1, 'edge': 1, 'num_class': 3},  # classification
    'imdb-binary': {'node': 1, 'edge': 1, 'num_class': 1},  # bin classification

    'ogbg-molesol': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-molbace': {'node': 9, 'edge': 3, 'num_class': 1},  # bin classification
    'ogbg-molhiv': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-moltox21': {'node': 9, 'edge': 3, 'num_class': 12},  # binary classification, but 12 tasks
    'qm9': {'node': 15, 'edge': 4, 'num_class': 1},  # regression, 13 labels, but we train 1 each split
    'ppgnqm9': {'node': 13, 'edge': 4, 'num_class': 1},  # regression, 13 labels, but we train 1 each split
    'exp': {'node': 2, 'edge': 0, 'num_class': 1},  # bin classification
    'csl': {'node': 1, 'edge': 1, 'num_class': 10},

    'sym_limits1': {'node': 4, 'edge': 0, 'num_class': 2},
    'sym_limits2': {'node': 4, 'edge': 0, 'num_class': 2},
    'sym_triangles': {'node': 1, 'edge': 0, 'num_class': 2},
    'sym_4cycles': {'node': 1, 'edge': 0, 'num_class': 2},
    'sym_skipcircles': {'node': 1, 'edge': 3, 'num_class': 10},
    'sym_lcc': {'node': 1, 'edge': 0, 'num_class': 3},

    'peptides-struct': {'node': 9, 'edge': 4, 'num_class': 11},  # regression, but 11 labels
    'peptides-func': {'node': 9, 'edge': 4, 'num_class': 10},  # 10-way classification
    'pcqm-contact': {'node': 9, 'edge': 3, 'num_class': None},  # edge prediction, but set it None!
    'coco-sp': {'node': 14, 'edge': 2, 'num_class': 81},
    'tree_2': {'node': 4, 'edge': 0, 'num_class': 4},
    'tree_3': {'node': 8, 'edge': 0, 'num_class': 8},
    'tree_4': {'node': 16, 'edge': 0, 'num_class': 16},
    'tree_5': {'node': 32, 'edge': 0, 'num_class': 32},
    'tree_6': {'node': 64, 'edge': 0, 'num_class': 64},
    'tree_7': {'node': 128, 'edge': 0, 'num_class': 128},
    'tree_8': {'node': 256, 'edge': 0, 'num_class': 256},

    # maybe we can make the labels const and add the n_labels here
    'leafcolor_2': {'node': 7, 'tree_depth': 2, 'n_leaf_labels': 2},
    'leafcolor_3': {'node': 15, 'tree_depth': 3, 'n_leaf_labels': 2},
    'leafcolor_4': {'node': 31, 'edge': 0, 'tree_depth': 4, 'n_leaf_labels': 2, 'num_class': 7},
    'leafcolor_5': {'node': 63, 'tree_depth': 5, 'n_leaf_labels': 2},
    'leafcolor_6': {'node': 127, 'tree_depth': 6, 'n_leaf_labels': 2},
    'leafcolor_7': {'node': 255, 'tree_depth': 7, 'n_leaf_labels': 2},
    'leafcolor_8': {'node': 511, 'tree_depth': 8, 'n_leaf_labels': 2},

    'cornell': {'node': 1703, 'edge': 0, 'num_class': 5},
    'texas': {'node': 1703, 'edge': 0, 'num_class': 5},
    'wisconsin': {'node': 1703, 'edge': 0, 'num_class': 5},

    'amazon-ratings': {'node': 300, 'edge': 0, 'num_class': 5},
}

TASK_TYPE_DICT = {
    'zinc': 'mae',
    'alchemy': 'mae',
    'proteins_full': 'acc',
    'mutag': 'acc',
    'ptc_mr': 'acc',
    'nci1': 'acc',
    'nci109': 'acc',
    'imdb-multi': 'acc',
    'imdb-binary': 'acc',
    'csl': 'acc',

    'peptides-struct': 'mae',
    'peptides-func': 'ap',
    'pcqm-contact': 'mrr_self_filtered',
    'coco-sp': 'f1_macro',

    'ogbg-molesol': 'rmse',
    'ogbg-molbace': 'rocauc',
    'ogbg-molhiv': 'rocauc',
    'ogbg-moltox21': 'rocauc',
    'qm9': 'mae',
    'ppgnqm9': 'mae',
    'exp': 'acc',
    'cornell': 'acc',
    'texas': 'acc',
    'wisconsin': 'acc',

    'amazon-ratings': 'acc',

    'tree_2': 'acc',
    'tree_3': 'acc',
    'tree_4': 'acc',
    'tree_5': 'acc',
    'tree_6': 'acc',
    'tree_7': 'acc',
    'tree_8': 'acc',

    'sym_limits1': 'acc',
    'sym_limits2': 'acc',
    'sym_triangles': 'acc',
    'sym_4cycles': 'acc',
    'sym_skipcircles': 'acc',
    'sym_lcc': 'acc',

    'leafcolor_2': 'acc',
    'leafcolor_3': 'acc',
    'leafcolor_4': 'acc',
    'leafcolor_5': 'acc',
    'leafcolor_6': 'acc',
    'leafcolor_7': 'acc',
    'leafcolor_8': 'acc',
}

CRITERION_DICT = {
    'zinc': nn.L1Loss(),
    'alchemy': nn.L1Loss(),
    'proteins_full': nn.BCEWithLogitsLoss(),
    'mutag': nn.BCEWithLogitsLoss(),
    'ptc_mr': nn.BCEWithLogitsLoss(),
    'nci1': nn.BCEWithLogitsLoss(),
    'nci109': nn.BCEWithLogitsLoss(),
    'imdb-multi': nn.CrossEntropyLoss(),
    'imdb-binary': nn.BCEWithLogitsLoss(),
    'csl': nn.CrossEntropyLoss(),

    'pcqm': nn.CrossEntropyLoss(),
    'peptides-struct': nn.L1Loss(),
    'peptides-func': nn.BCEWithLogitsLoss(),
    'pcqm-contact': nn.BCEWithLogitsLoss(),
    'coco-sp': weighted_cross_entropy,
    'ogbg-molesol': nn.MSELoss(),
    'ogbg-molbace': nn.BCEWithLogitsLoss(),
    'ogbg-molhiv': nn.BCEWithLogitsLoss(),
    'ogbg-moltox21': nn.BCEWithLogitsLoss(),
    'qm9': nn.MSELoss(),
    'ppgnqm9': nn.MSELoss(),
    'exp': nn.BCEWithLogitsLoss(),

    'tree_2': nn.CrossEntropyLoss(),
    'tree_3': nn.CrossEntropyLoss(),
    'tree_4': nn.CrossEntropyLoss(),
    'tree_5': nn.CrossEntropyLoss(),
    'tree_6': nn.CrossEntropyLoss(),
    'tree_7': nn.CrossEntropyLoss(),
    'tree_8': nn.CrossEntropyLoss(),

    'leafcolor_2': nn.CrossEntropyLoss(),
    'leafcolor_3': nn.CrossEntropyLoss(),
    'leafcolor_4': nn.CrossEntropyLoss(),
    'leafcolor_5': nn.CrossEntropyLoss(),
    'leafcolor_6': nn.CrossEntropyLoss(),
    'leafcolor_7': nn.CrossEntropyLoss(),
    'leafcolor_8': nn.CrossEntropyLoss(),

    'sym_limits1': nn.CrossEntropyLoss(),
    'sym_limits2': nn.CrossEntropyLoss(),
    'sym_triangles': nn.CrossEntropyLoss(),
    'sym_4cycles': nn.CrossEntropyLoss(),
    'sym_skipcircles': nn.CrossEntropyLoss(),
    'sym_lcc': nn.CrossEntropyLoss(),

    'cornell': nn.CrossEntropyLoss(),
    'texas': nn.CrossEntropyLoss(),
    'wisconsin': nn.CrossEntropyLoss(),
    'amazon-ratings': nn.CrossEntropyLoss(),
}

SCHEDULER_MODE = {
    'acc': 'max',
    'mae': 'min',
    'mse': 'min',
    'rocauc': 'max',
    'rmse': 'min',
    'ap': 'max',
    'mrr': 'max',
    'mrr_self_filtered': 'max',
    'f1_macro': 'max',
}

ENCODER_TYPE_DICT = {
    'zinc': {'bond': 'zinc', 'atom': 'zinc'},
    'peptides-func': {'bond': 'ogb', 'atom': 'ogb'},
    'peptides-struct': {'bond': 'ogb', 'atom': 'ogb'},
    'pcqm-contact': {'bond': 'ogb', 'atom': 'ogb'},
    'ogbg-molhiv': {'bond': 'ogb', 'atom': 'ogb'},
    'coco-sp': {'bond': 'coco', 'atom': 'coco'},
    'cornell': {'bond': None, 'atom': 'linear'},
    'amazon-ratings': {'bond': None, 'atom': 'linear'},
    'csl': {'bond': 'linear', 'atom': 'linear'},
    'exp': {'bond': None, 'atom': 'exp'},
    'proteins_full': {'bond': None, 'atom': 'linear'},
    'nci1': {'bond': None, 'atom': 'linear'},
    'nci109': {'bond': None, 'atom': 'linear'},
    'ptc_mr': {'bond': 'linear', 'atom': 'linear'},
    'mutag': {'bond': 'linear', 'atom': 'linear'},
    'imdb-multi': {'bond': 'linear', 'atom': 'linear'},
    'imdb-binary': {'bond': 'linear', 'atom': 'linear'},
    'qm9': {'bond': 'linear', 'atom': 'linear'},
    'alchemy': {'bond': 'linear', 'atom': 'linear'},
    'tree_2': {'bond': None, 'atom': 'bi_embedding'},
    'tree_3': {'bond': None, 'atom': 'bi_embedding'},
    'tree_4': {'bond': None, 'atom': 'bi_embedding'},
    'tree_5': {'bond': None, 'atom': 'bi_embedding'},
    'tree_6': {'bond': None, 'atom': 'bi_embedding'},
    'tree_7': {'bond': None, 'atom': 'bi_embedding'},
    'tree_8': {'bond': None, 'atom': 'bi_embedding'},
    'leafcolor_2': {'bond': None, 'atom': 'bi_embedding_cat'},
    'leafcolor_3': {'bond': None, 'atom': 'bi_embedding_cat'},
    'leafcolor_4': {'bond': None, 'atom': 'bi_embedding_cat'},
    'leafcolor_5': {'bond': None, 'atom': 'bi_embedding_cat'},
    'leafcolor_6': {'bond': None, 'atom': 'bi_embedding_cat'},
    'leafcolor_7': {'bond': None, 'atom': 'bi_embedding_cat'},
    'leafcolor_8': {'bond': None, 'atom': 'bi_embedding_cat'},
}

# whether to set cached=True for GCNConv, only True if transductive
GCN_CACHE = {
    'zinc': False,
    'peptides-func': False,
    'peptides-struct': False,
    'cornell': True,
    'amazon-ratings': True,
    'csl': False,
    'exp': False,
}
