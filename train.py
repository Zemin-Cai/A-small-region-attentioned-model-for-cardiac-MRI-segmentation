import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TORCH_HOME'] ='/media/stu/74A84304A842C478/ConvNextUnet/pretrained_ckpt'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from network.network import ConvNextUNet
from trainer import trainer_acdc
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='root2/data/ACDC', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./root2/lists/lists_ACDC', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir', default='snapshot_path/')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')  # 24
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')  # 1
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,  # 0.01
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "ACDC":
    args.root_path = os.path.join(args.root_path, "train_npz") # train_npz
# from configs.other_config import get_config
# config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'ACDC': {
            'root_path': args.root_path,
            'list_dir': './root2/lists/lists_ACDC',
            'num_classes': 4,
        },
    }

    # if args.batch_size != 24 and args.batch_size % 6 == 0:
    #     args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = ConvNextUNet(num_classes=args.num_classes).cuda()



    trainer = {'ACDC': trainer_acdc,}
    trainer[dataset_name](args, net, args.output_dir)
