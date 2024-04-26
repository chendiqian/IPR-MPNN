from functools import partial

import torch.nn as nn
from torch_geometric.nn import MLP

from data.const import DATASET_FEATURE_STAT_DICT, ENCODER_TYPE_DICT
from models.base2centroid import GNNMultiEdgeset
from models.hetero_gnn import HeteroGNN
from models.hybrid_model import HybridModel
from models.my_encoders import get_bond_encoder, get_atom_encoder
from models.nn_utils import get_graph_pooling, inter_ensemble_pooling, jumping_knowledge
from models.prediction_head import Predictor


def get_model(args, device):
    # get atom encoder and bond encoder
    def get_atom_encoder_handler():
        return get_atom_encoder(ENCODER_TYPE_DICT[args.dataset.lower()]['atom'],
                                args.hetero.hidden if hasattr(args, 'hetero') else args.gnn.hidden,  # plain GNN
                                DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                args.encoder.lap if hasattr(args.encoder, 'lap') else None,
                                args.encoder.rwse if hasattr(args.encoder, 'rwse') else None)

    def get_bond_encoder_handler():
        return get_bond_encoder(ENCODER_TYPE_DICT[args.dataset.lower()]['bond'],
                                args.hetero.hidden if hasattr(args, 'hetero') else args.gnn.hidden,  # plain GNN
                                DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],)

    # base to centroid
    if hasattr(args, 'base2centroid') and args.base2centroid is not None:
        base2centroid_model = GNNMultiEdgeset(
            conv=args.base2centroid.conv,
            centroid_aggr=args.base2centroid.centroid_aggr,
            atom_encoder_handler=get_atom_encoder_handler,
            bond_encoder_handler=get_bond_encoder_handler,
            hidden=args.hetero.hidden,
            num_conv_layers=args.base2centroid.num_conv_layers,
            num_mlp_layers=args.base2centroid.num_mlp_layers,
            out_feature=args.hetero.cent_hidden if hasattr(args.hetero, 'cent_hidden') else args.hetero.hidden,
            norm=args.base2centroid.norm,
            activation=args.base2centroid.activation,
            dropout=args.base2centroid.dropout,
        )
    else:
        base2centroid_model = None

    # heterogeneous, hierarchical GNN
    if hasattr(args, 'hetero') and args.hetero is not None:
        hetero_mpnn = HeteroGNN(
            conv=args.hetero.conv,
            b2c_conv=args.hetero.b2c if hasattr(args.hetero, 'b2c') else args.hetero.conv,
            c2b_conv=args.hetero.c2b if hasattr(args.hetero, 'c2b') else args.hetero.conv,
            c2c_conv=args.hetero.c2c if hasattr(args.hetero, 'c2c') else args.hetero.conv,
            atom_encoder_handler=get_atom_encoder_handler,
            bond_encoder_handler=get_bond_encoder_handler,
            hid_dim=args.hetero.hidden,
            centroid_hid_dim=args.hetero.cent_hidden if hasattr(args.hetero, 'cent_hidden') else args.hetero.hidden,
            num_conv_layers=args.hetero.num_conv_layers,
            num_mlp_layers=args.hetero.num_mlp_layers,
            dropout=args.hetero.dropout,
            norm=args.hetero.norm,
            activation=args.hetero.activation,
            use_res=args.hetero.residual,
            delay=args.hetero.delay if hasattr(args.hetero, 'delay') else 0,
            aggr=args.hetero.aggr,
            parallel=args.hetero.parallel,
        )
    else:
        hetero_mpnn = None

    if base2centroid_model is not None and hetero_mpnn is not None:
        # Todo: enable this
        if args.hybrid_model.intra_graph_pool in ['root', 'edge']:  # node / edge prediction
            assert args.hybrid_model.target == 'base', "Unable to use centroids"
        intra_graph_pool_func, intra_graph_pool_attr = get_graph_pooling(args.hybrid_model.intra_graph_pool)

        def get_prediction_head():  # a wrapper func
            return Predictor(
                pred_target=args.hybrid_model.target,
                inter_ensemble_pool=partial(inter_ensemble_pooling,
                                            inter_pool=args.hybrid_model.inter_ensemble_pool),
                inter_base_pred_head=MLP(
                    in_channels=-1,
                    hidden_channels=args.hetero.hidden,
                    out_channels=args.hetero.hidden,
                    num_layers=args.hybrid_model.inter_pred_layer,
                    norm=None) if args.hybrid_model.inter_pred_layer > 0 else nn.Identity(),
                inter_cent_pred_head=MLP(
                    in_channels=-1,
                    hidden_channels=args.hetero.hidden,
                    out_channels=args.hetero.hidden,
                    num_layers=args.hybrid_model.inter_pred_layer,
                    norm=None) if args.hybrid_model.inter_pred_layer > 0 else nn.Identity(),
                intra_graph_pool=intra_graph_pool_func,
                intra_pred_head=MLP(
                    in_channels=-1,
                    hidden_channels=args.hetero.hidden,
                    out_channels=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'] if
                    DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'] is not None else args.hetero.hidden,
                    num_layers=args.hybrid_model.intra_pred_layer,
                    norm=None)
            )

        if args.hybrid_model.jk == 'identity':
            # we keep the intermediate embeddings un-aggregated, and need a head per tensor
            pred_head = nn.ModuleList([get_prediction_head() for _ in range(args.hetero.num_conv_layers)])
        else:
            pred_head = get_prediction_head()

        hybrid_model = HybridModel(
            base2centroid_model=base2centroid_model,
            assignment=args.base2centroid.assignment,
            hetero_gnn=hetero_mpnn,

            jk_func=partial(jumping_knowledge, jk=args.hybrid_model.jk),
            graph_pool_idx=intra_graph_pool_attr,
            pred_head=pred_head,
        ).to(device)
        return hybrid_model
    else:
        # normal GNN
        raise NotImplementedError
