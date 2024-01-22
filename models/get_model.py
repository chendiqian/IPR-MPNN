import torch.nn as nn

from torch_geometric.nn import MLP

from data.const import DATASET_FEATURE_STAT_DICT, ENCODER_TYPE_DICT, GCN_CACHE
from models.base2centroid import GNNMultiEdgeset
from models.hetero_gnn import HeteroGNN
from models.hybrid_model import HybridModel
from models.my_encoders import get_bond_encoder, get_atom_encoder
from models.scorer_model import ScorerGNN
from samplers.get_sampler import get_sampler


def get_model(args, device):
    # get atom encoder and bond encoder
    def get_atom_encoder_handler(partition_encoder = False):
        return get_atom_encoder(ENCODER_TYPE_DICT[args.dataset.lower()]['atom'],
                                args.hetero.hidden,
                                DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'],
                                args.encoder.lap if hasattr(args.encoder, 'lap') else None,
                                args.encoder.rwse if hasattr(args.encoder, 'rwse') else None,
                                args.encoder.partition if (hasattr(args.encoder, 'partition') and partition_encoder) else None)

    def get_bond_encoder_handler():
        return get_bond_encoder(ENCODER_TYPE_DICT[args.dataset.lower()]['bond'],
                                args.hetero.hidden,
                                DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['edge'],)

    # scorer model
    if hasattr(args, 'scorer_model') and args.scorer_model is not None:
        scorer_model = ScorerGNN(
            conv=args.scorer_model.conv,
            conv_cache=GCN_CACHE[args.dataset.lower()],
            atom_encoder_handler=get_atom_encoder_handler,
            bond_encoder_handler=get_bond_encoder_handler,
            in_feature=args.hetero.hidden,
            hidden=args.scorer_model.hidden,
            num_conv_layers=args.scorer_model.num_conv_layers,
            num_mlp_layers=args.scorer_model.num_mlp_layers,
            max_num_centroids=max(args.scorer_model.num_centroids),  # this is a list
            num_ensemble=args.sampler.num_ensemble,
            norm=args.scorer_model.norm,
            activation=args.scorer_model.activation,
            dropout=args.scorer_model.dropout
        )
    else:
        scorer_model = None

    # base to centroid
    if hasattr(args, 'base2centroid') and args.base2centroid is not None:
        base2centroid_model = GNNMultiEdgeset(
            conv=args.base2centroid.conv,
            centroid_aggr=args.base2centroid.centroid_aggr,
            atom_encoder_handler=get_atom_encoder_handler,
            bond_encoder_handler=get_bond_encoder_handler,
            hidden=args.hetero.hidden,
            centroid_hid_dim=args.hetero.cent_hidden if hasattr(args.hetero, 'cent_hidden') else args.hetero.hidden,
            num_conv_layers=args.base2centroid.num_conv_layers,
            num_mlp_layers=args.base2centroid.num_mlp_layers,
            out_feature=args.hetero.hidden,
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

    # discrete sampler
    if hasattr(args, 'sampler') and args.sampler is not None:
        sampler = get_sampler(args.sampler, device)
    else:
        sampler = None

    if scorer_model is not None and \
            base2centroid_model is not None and \
            hetero_mpnn is not None and \
            sampler is not None:

        intra_pred_head = nn.ModuleList([MLP(in_channels=-1,
                              hidden_channels=args.hetero.hidden,
                              out_channels=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'],
                              num_layers=args.hybrid_model.intra_pred_layer,
                              norm=None) for _ in range(args.hetero.num_conv_layers)]) if args.hybrid_model.intermediate_heads else MLP(in_channels=-1,
                              hidden_channels=args.hetero.hidden,
                              out_channels=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'],
                              num_layers=args.hybrid_model.intra_pred_layer,
                              norm=None)
        
        inter_base_pred_head = MLP(
            in_channels=-1,
            hidden_channels=args.hetero.hidden,
            out_channels=args.hetero.hidden,
            num_layers=args.hybrid_model.inter_pred_layer,
            norm=None)
        inter_cent_pred_head = MLP(
            in_channels=-1,
            hidden_channels=args.hetero.hidden,
            out_channels=args.hetero.hidden,
            num_layers=args.hybrid_model.inter_pred_layer,
            norm=None)

        hybrid_model = HybridModel(
            device=device,
            scorer_model=scorer_model,
            list_num_centroids=args.scorer_model.num_centroids,  # this is a list
            base2centroid_model=base2centroid_model,
            sampler=sampler,
            hetero_gnn=hetero_mpnn,

            jk=args.hybrid_model.jk,
            target=args.hybrid_model.target,
            intra_pred_head=intra_pred_head,
            inter_base_pred_head=inter_base_pred_head,
            inter_cent_pred_head=inter_cent_pred_head,
            intra_graph_pool=args.hybrid_model.intra_graph_pool,
            inter_ensemble_pool=args.hybrid_model.inter_ensemble_pool,
            auxloss_dict=args.auxloss if hasattr(args, 'auxloss') and args.auxloss is not None else None
        ).to(device)
        return hybrid_model
    else:
        # normal GNN
        raise NotImplementedError
