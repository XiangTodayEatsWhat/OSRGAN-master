import argparse
from utils import str2bool

parser = argparse.ArgumentParser(description='OSRGAN')

parser.add_argument('--name', default='OSRGAN-dSprites', type=str, help='name of the experiment')
parser.add_argument('--model_type', default='OBE', type=str, help='model_type, OBE or DCT')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--z_dim', default=60, type=int, help='dimension of the representation z')
parser.add_argument("--code_dim", type=int, default=5, help="dimension of the code c")
parser.add_argument("--n_classes", type=int, default=6, help="num for visual")
parser.add_argument("--seed", type=int, default=7, help="random seed")

parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
parser.add_argument('--G_iteration', type=int, default=4, help='times of training G')
parser.add_argument('--lr', default=9e-4, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1 parameter of the Adam optimizer')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
parser.add_argument('--g_weight', default=10, type=float, help='g_weight of g loss')
parser.add_argument('--info_weight', default=1, type=float, help='info_weight of g loss')
parser.add_argument('--con_weight', default=1, type=float, help='con_weight of g loss')
parser.add_argument('--metric_times', default=1, type=int, help='times of count metric')
parser.add_argument('--dset_dir', default='../data', type=str, help='dataset directory')
parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')

parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')
parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
parser.add_argument('--viz_ll_iter', default=100, type=int, help='visdom line data logging iter')
parser.add_argument('--viz_la_iter', default=500, type=int, help='visdom line data applying iter')
parser.add_argument('--viz_vc_iter', default=1000, type=int, help='visdom image data applying iter')

parser.add_argument('--ckpt_dir', default='./outputs/checkpoints', type=str, help='checkpoint directory')
parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')

parser.add_argument('--metric_log_dir', default='./outputs/metric', type=str, help='metric log directory')

opt = parser.parse_args()

opt.name =  '{}-{}-{}'.format(opt.name, opt.seed, opt.model_type)