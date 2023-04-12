import os
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from augmentations.cut_mix_up import get_cutmix_and_mixup_output_and_loss


def train_one_epoch(epoch, vis, train_loader, model, optimizer, criterion, scheduler, opts):
    print('Training of epoch [{}]'.format(epoch))

    model.train()
    tic = time.time()

    for i, data in enumerate(tqdm(train_loader)):

        # ----------------- cuda -----------------
        images = data[0].to(int(opts.gpu_ids[opts.rank]))
        labels = data[1].to(int(opts.gpu_ids[opts.rank]))

        # ----------------- loss -----------------
        if opts.is_vit_data_augmentation:
            outputs, loss = get_cutmix_and_mixup_output_and_loss(images, labels, criterion, model, opts)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # ----------------- update -----------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # get lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # time
        toc = time.time()

        # visualization
        if i % opts.vis_step == 0 and opts.rank == 0:
            print('Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, lr: {5:.5f}, Time: {6:.2f}'.format(epoch,
                                                                                                      opts.epoch,
                                                                                                      i,
                                                                                                      len(train_loader),
                                                                                                      loss.item(),
                                                                                                      lr,
                                                                                                      toc - tic))

            vis.line(X=torch.ones((1, 1)) * i + epoch * len(train_loader),
                     Y=torch.Tensor([loss]).unsqueeze(0),
                     update='append',
                     win='train_loss_for_{}'.format(opts.name),
                     opts=dict(x_label='step',
                               y_label='loss',
                               title='train loss for {}'.format(opts.name),
                               legend=['train_loss']))

    # save pth file
    if opts.rank == 0:

        save_path = os.path.join(opts.log_dir, opts.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        if epoch > opts.save_step:
            torch.save(checkpoint, os.path.join(save_path, opts.name + '.{}.pth.tar'.format(epoch)))
            print("save .pth")
