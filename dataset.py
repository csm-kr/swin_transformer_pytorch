import torch
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from augmentations.auto_aug import CIFAR10Policy, ImageNetPolicy
from augmentations.random_erasing import RandomErasing
from augmentations.sampler import RASampler
from augmentations.sampler_imagenet import RASampler as RASampler_imagenet
from torch.utils.data.distributed import DistributedSampler


def build_dataloader(opts, is_return_mean_std=False):

    train_loader = None
    test_loader = None
    MEAN = None
    STD = None

    if opts.data_type == 'cifar10':
        print('dataset : {}'.format(opts.data_type))

        opts.num_classes = 10
        opts.img_size = 32
        opts.data_root = './data/CIFAR10'
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

        if opts.is_vit_data_augmentation:
            transform_train = tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                CIFAR10Policy(),
                tfs.ToTensor(),
                tfs.Normalize(mean=MEAN, std=STD),
                RandomErasing(probability=0.25, sl=0.02, sh=0.4, r1=0.3, mean=MEAN)
            ])
        else:
            transform_train = tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize(mean=MEAN, std=STD),
            ])

        transform_test = tfs.Compose([tfs.ToTensor(),
                                      tfs.Normalize(mean=MEAN,
                                                    std=STD),
                                      ])

        train_set = CIFAR10(root=opts.data_root,
                            train=True,
                            download=True,
                            transform=transform_train)

        test_set = CIFAR10(root=opts.data_root,
                           train=False,
                           download=True,
                           transform=transform_test)
        if opts.is_vit_data_augmentation:
            train_loader = DataLoader(train_set,
                                      num_workers=opts.num_workers,
                                      batch_sampler=RASampler(dataset_len=len(train_set), batch_size=opts.batch_size,
                                                              repetitions=1, len_factor=3., shuffle=True, drop_last=False),
                                      pin_memory=True,
                                      )
        else:
            train_loader = DataLoader(train_set,
                                      batch_size=opts.batch_size,
                                      shuffle=True,
                                      num_workers=opts.num_workers,
                                      pin_memory=True,
                                      )

        test_loader = DataLoader(test_set,
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.num_workers,
                                 pin_memory=True,
                                 )

        if is_return_mean_std:
            return train_loader, test_loader, MEAN, STD

    elif opts.data_type == 'cifar100':
        print('dataset : {}'.format(opts.data_type))

        opts.num_classes = 100
        opts.img_size = 32
        opts.data_root = './data/CIFAR100'
        MEAN, STD = (0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

        if opts.is_vit_data_augmentation:
            transform_train = tfs.Compose([
                tfs.RandomHorizontalFlip(),
                tfs.RandomCrop(32, padding=4),
                CIFAR10Policy(),
                tfs.ToTensor(),
                tfs.Normalize(mean=MEAN, std=STD),
                RandomErasing(probability=0.25, sl=0.02, sh=0.4, r1=0.3, mean=MEAN)
            ])
        else:
            transform_train = tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize(mean=MEAN, std=STD),
            ])

        transform_test = tfs.Compose([tfs.ToTensor(),
                                      tfs.Normalize(mean=MEAN,
                                                    std=STD),
                                      ])

        train_set = CIFAR100(root=opts.data_root,
                             train=True,
                             download=True,
                             transform=transform_train)

        test_set = CIFAR100(root=opts.data_root,
                            train=False,
                            download=True,
                            transform=transform_test)
        if opts.is_vit_data_augmentation:
            train_loader = DataLoader(train_set,
                                      num_workers=opts.num_workers,
                                      batch_sampler=RASampler(dataset_len=len(train_set), batch_size=opts.batch_size,
                                                              repetitions=1, len_factor=3., shuffle=True,
                                                              drop_last=False),
                                      pin_memory=True,
                                      )
        else:
            train_loader = DataLoader(train_set,
                                      batch_size=opts.batch_size,
                                      shuffle=True,
                                      num_workers=opts.num_workers,
                                      pin_memory=True,
                                      )

        test_loader = DataLoader(test_set,
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.num_workers,
                                 pin_memory=True,
                                 )

        if is_return_mean_std:
            return train_loader, test_loader, MEAN, STD

    elif opts.data_type == 'imagenet':
        print('dataset : {}'.format(opts.data_type))
        opts.num_classes = 1000
        # opts.data_root = '/home/cvmlserver7/Sungmin/data/imagenet'
        opts.input_size = 224
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if opts.is_vit_data_augmentation:
            transform_train = tfs.Compose([
                tfs.Resize(256),
                tfs.RandomCrop(224, padding=4),
                tfs.RandomHorizontalFlip(),
                ImageNetPolicy(),
                tfs.ToTensor(),
                tfs.Normalize(mean=MEAN, std=STD),
                RandomErasing(probability=0.25, sl=0.02, sh=0.4, r1=0.3, mean=MEAN)
            ])

            transform_test = tfs.Compose([tfs.Resize(opts.input_size),
                                          tfs.CenterCrop(224),
                                          tfs.ToTensor(),
                                          tfs.Normalize(mean=MEAN,
                                                        std=STD),
                                          ])
            train_set = ImageNet(root=opts.data_root, transform=transform_train, split='train')
            test_set = ImageNet(root=opts.data_root, transform=transform_test, split='val')

            train_sampler = RASampler_imagenet(train_set, num_replicas=opts.world_size, rank=opts.rank, shuffle=True)
            test_sampler = DistributedSampler(test_set, num_replicas=opts.world_size, rank=opts.rank, shuffle=False)

            train_loader = DataLoader(dataset=train_set,
                                      batch_size=int(opts.batch_size / opts.world_size),
                                      shuffle=False,
                                      num_workers=int(opts.num_workers / opts.world_size),
                                      sampler=train_sampler,
                                      pin_memory=True)

            test_loader = DataLoader(dataset=test_set,
                                     batch_size=int(opts.batch_size / opts.world_size),
                                     shuffle=False,
                                     num_workers=int(opts.num_workers / opts.world_size),
                                     sampler=test_sampler,
                                     pin_memory=True)

            return train_loader, test_loader, train_sampler

    return train_loader,