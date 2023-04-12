import configargparse


def get_args_parser():
    parser = configargparse.ArgumentParser(add_help=False)
    # config
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str)

    # vis
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--vis_step', type=int, default=100)

    # data
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--num_classes', type=int)

    # data augmentation
    parser.set_defaults(is_vit_data_augmentation=False)  # auto aug, random erasing, label smoothing, cutmix, mixup,
    parser.add_argument('--mixup_beta', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--vit_data_augmentation_true', dest='is_vit_data_augmentation', action='store_true')

    # model
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--in_chans', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)
    # qkv_bias false, drop_rate 0.(proj, mlp...) , atte_drop_rate 0. (only attention)
    parser.add_argument('--drop_path', type=float)
    parser.set_defaults(has_cls_token=True)
    parser.add_argument('--has_cls_token_false', dest='has_cls_token', action='store_false')
    parser.set_defaults(has_last_norm=True)
    parser.add_argument('--has_last_norm_false', dest='has_last_norm', action='store_false')
    parser.set_defaults(has_basic_poe=True)
    parser.add_argument('--has_basic_poe_false', dest='has_basic_poe', action='store_false', help='if not, sinusoid 2d')

    parser.set_defaults(has_auto_encoder=False)
    parser.add_argument('--has_auto_encoder_true', dest='has_auto_encoder', action='store_true')
    parser.set_defaults(use_sasa=False)
    parser.add_argument('--use_sasa_true', dest='use_sasa', action='store_true')
    parser.set_defaults(use_gpsa=False)
    parser.add_argument('--use_gpsa_true', dest='use_gpsa', action='store_true')

    # train & optimizer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--warmup', type=int, help='warmup epoch')
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--save_step', type=int, default=1000, help='if save_step < epoch, then save')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # distributed
    parser.set_defaults(distributed=False)
    parser.add_argument('--gpu_ids', nargs="+")
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world_size', type=int)

    # test
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--test_epoch', type=str, default='best')
    return parser

