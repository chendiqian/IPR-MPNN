from data.const import DATASET_FEATURE_STAT_DICT
from models.base2centroid import GNNMultiEdgeset
from models.hetero_gnn import HeteroGNN
from models.hybrid_model import HybridModel
from models.my_encoders import get_bond_encoder, get_atom_encoder
from models.scorer_model import ScorerGNN
from samplers.get_sampler import get_sampler
from torch_geometric.nn import MLP


def get_model(args, device):
    # get atom encoder and bond encoder
    atom_encoder = get_atom_encoder(args.encoder.atom,
                                    args.hetero.hidden,
                                    DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['node'])
    bond_encoder = get_bond_encoder(args.encoder.bond, args.hetero.hidden)

    # scorer model
    if hasattr(args, 'scorer_model') and args.scorer_model is not None:
        scorer_model = ScorerGNN(
            conv=args.scorer_model.conv,
            bond_encoder=bond_encoder,
            in_feature=args.hetero.hidden,
            hidden=args.scorer_model.hidden,
            num_conv_layers=args.scorer_model.num_conv_layers,
            num_mlp_layers=args.scorer_model.num_mlp_layers,
            num_centroids=args.scorer_model.num_centroids,
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
            edge_encoder=bond_encoder,
            hidden=args.hetero.hidden,
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
            edge_encoder=bond_encoder,
            hid_dim=args.hetero.hidden,
            num_conv_layers=args.hetero.num_conv_layers,
            num_mlp_layers=args.hetero.num_mlp_layers,
            dropout=args.hetero.dropout,
            norm=args.hetero.norm,
            activation=args.hetero.activation,
            use_res=args.hetero.residual,
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

        intra_pred_head = MLP(in_channels=args.hetero.hidden,
                              hidden_channels=args.hetero.hidden,
                              out_channels=DATASET_FEATURE_STAT_DICT[args.dataset.lower()]['num_class'],
                              num_layers=args.hybrid_model.intra_pred_layer,
                              norm=None)
        inter_pred_head = MLP(in_channels=args.hetero.hidden,
                              hidden_channels=args.hetero.hidden,
                              out_channels=args.hetero.hidden,
                              num_layers=args.hybrid_model.inter_pred_layer,
                              norm=None)

        hybrid_model = HybridModel(
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
            scorer_model=scorer_model,
            base2centroid_model=base2centroid_model,
            sampler=sampler,
            hetero_gnn=hetero_mpnn,

            target=args.hybrid_model.target,
            intra_pred_head=intra_pred_head,
            inter_pred_head=inter_pred_head,
            intra_graph_pool=args.hybrid_model.intra_graph_pool,
            inter_ensemble_pool=args.hybrid_model.inter_ensemble_pool,
        ).to(device)
        return hybrid_model
    else:
        # normal GNN
        raise NotImplementedError
