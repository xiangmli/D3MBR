import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--cuda', default=True, help="use cuda or not")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--epoch', default=25, type=int, help="max epoch number")
    parser.add_argument('--early_stop', default=True, type=bool, help="max epoch number")
    parser.add_argument('--stop_epochs', default=3, type=int, help="max epoch number")


    parser.add_argument('--dataset', default='IJCAI', type=str, help='name of dataset: IJCAI_15, Tmall')
    parser.add_argument('--batch_size', type=int, default=2048, help="the batch size for bpr loss training procedure")
    parser.add_argument('--aux_batch_size', type=int, default=512, help="the batch size for auxiliary behavior")
    parser.add_argument('--dropout', type=float, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")

    # rec params
    parser.add_argument("--num_ng", type=int, default=1, help="sample negative items for training")
    parser.add_argument("--latdim", type=int, default=32, help="dim of user and item embeddings")
    parser.add_argument("--n_layer", type=int, default=2, help="num of gcn layers")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument('--lr', type=float, default=0.0001, help="the learning rate")
    # target behavior reverse DNN params
    parser.add_argument('--dims', type=str, default='[200,600]', help='the dims for the DNN')
    parser.add_argument('--aux_dims', type=str, default='[300]', help='the dims for the auxiliary behavior DNN')
    parser.add_argument('--act', type=str, default='tanh', help='the activate function for the DNN')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--diff_lr', type=float, default=0.0001, help="the learning rate")
    parser.add_argument('--aux_lr', type=float, default=0.0001, help="the learning rate")

    # diff params
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=2, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.005, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.005, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=2, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True,
                        help='assign different weight to different timestep or not')

    parser.add_argument('--alpha', type=float, default=0.1, help='balance rec loss and reconstruct loss')

    parser.add_argument('--beta', type=float, help='Exponent for weights adjusting', default=0.2)

    parser.add_argument('--keep_target_rate', type=int, default=0.8,
                        help='The percentage of target behavior data to retain for guiding auxiliary behavior denoising')

    parser.add_argument('--keep_rate', type=int, default=0.8, help='the keep rate of interactions in the denoised graph')

    parser.add_argument('--restrict', type=bool, default=True,
                        help='If True, retain only the high-score items from the original interactions in the denoised graph. '
                             'If False, select high-score items from all available items in the denoised graph.')

    return parser.parse_args()

args = parse_args()
