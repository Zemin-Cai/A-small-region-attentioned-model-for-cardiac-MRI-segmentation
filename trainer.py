import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import logging
import random
import sys
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer_acdc(args, model, snapshot_path):
    from dataset.dataset_acdc import ACDC_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))


    print("The length of train set is: {}".format(len(db_train)))
    # print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,  # 4
    #                          worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)  # .cuda()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # detail_loss_func = DetailAggregateLoss()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)  # 0.0001
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0  # 0

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        mean_dice = 0.0
        print("Start train:")
        total_loss = 0.0
        batch = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            batch += 1
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # outputs = model(image_batch)
            outputs = model(image_batch)
            # print(outputs.shape)
            # print(label_batch[:].long().shape)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            dice = 1 - loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            mean_dice += dice
            mean = mean_dice / (i_batch + 1)
            total_loss += loss.item()
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/dice', dice, iter_num)
            writer.add_scalar('info/mean_dice', mean, iter_num)

            logging.info('iteration %d : loss : %f, mean_dice: %f' % (
                iter_num, loss.item(), mean.item()))
            logging.info('iteration %d : loss : %f, loss_ce: %f, dice: %f, mean_dice: %f' % (
                iter_num, loss.item(), loss_ce.item(), dice.item(), mean.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # outputs = torch.argmax(torch.sigmoid(outputs), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        epoch_loss = total_loss / batch
        writer.add_scalar('info/epoch_loss', epoch_loss, epoch_num)

        # print("Start Validation:")
        # val_epoch_loss, val_epoch_loss_ce, val_epoch_dice = validation(args, valloader, model)
        # logging.info('val_loss : %f, val_loss_ce: %f, val_dice: %f' % (val_epoch_loss, val_epoch_loss_ce,
        #                                                                val_epoch_dice))
        # writer.add_scalar('info/epoch_val_loss', val_epoch_loss, epoch_num)
        writer.add_scalars('train_loss_and_val_loss', {'info/train_loss': epoch_loss}#,
                                                       # 'info/val_loss': val_epoch_loss}
                           , epoch_num)

        save_interval = 10
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

# def validation(args, valloader, model):
#     model.eval()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(args.num_classes)
#     detail_loss_func = DetailAggregateLoss()
#     val_total_dice = 0.0
#     val_total_loss = 0.0
#     val_total_loss_ce = 0.0
#     iteration = 0
#
#     with torch.no_grad():
#         for i_batch, sampled_batch in enumerate(valloader):
#             iteration += 1
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#
#             preds = model(image_batch)
#             val_loss_ce = ce_loss(preds, label_batch[:].long())
#             val_loss_dice = dice_loss(preds, label_batch, softmax=True)
#             val_loss = 0.4 * val_loss_ce + 0.6 * val_loss_dice
#             val_dice = 1 - val_loss_dice
#
#             val_total_loss += val_loss.item()
#             val_total_loss_ce += val_loss_ce.item()
#             val_total_dice += val_dice.item()
#
#         val_mean_loss = val_total_loss / iteration
#         val_mean_loss_ce = val_total_loss_ce / iteration
#         val_mean_dice = val_total_dice / iteration
#     return val_mean_loss, val_mean_loss_ce, val_mean_dice