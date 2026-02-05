import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--drug_emb_path', type=str, required=True)
    parser.add_argument('--drug_out_dim', type=int, default=256)

    parser.add_argument('--cell_exp_dim', type=int, required=True)
    parser.add_argument('--cell_path_dim', type=int, required=True)
    parser.add_argument('--cell_out_dim', type=int, default=256)

    parser.add_argument('--ban_heads', type=int, default=2)
    parser.add_argument('--ban_dropout', type=float, default=0.5)
    parser.add_argument('--ban_k', type=int, default=3)

    parser.add_argument('--mlp_in_dim', type=int, default=256)
    parser.add_argument('--mlp_hidden_dim', type=int, nargs='+',
                        default=[512, 256])

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    return parser.parse_args()
