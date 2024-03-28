import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk



import argparse
import torch.nn as nn
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):  # patch_size=[256, 256],  try:patch_size=[224, 224]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    repetitions = image.shape[1]

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        # prediction_a1 = np.zeros((label.shape[0], 14, 14))
        # prediction_a2 = np.zeros((label.shape[0], 28, 28))
        # prediction_a3 = np.zeros((label.shape[0], 56, 56))
        # prediction_a4 = np.zeros((label.shape[0], 56, 56))
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # out = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                # a11 = a1.cpu().detach().numpy()
                # a21 = a2.cpu().detach().numpy()
                # a31 = a3.cpu().detach().numpy()
                # a41 = a4.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 0++++++++++++++++
                    # pred_a1 = zoom(a11, (14, 14), order=0)
                    # pred_a2 = zoom(a21, (28, 28), order=0)
                    # pred_a3 = zoom(a31, (x / patch_size[0], y / patch_size[1]), order=0)
                    # pred_a4 = zoom(a41, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                    # pred_a1 = a11
                    # pred_a2 = a21
                    # pred_a3 = a31
                    # pred_a4 = a41
                prediction[ind] = pred
                # prediction_a1[ind] = pred_a1
                # prediction_a2[ind] = pred_a2
                # prediction_a3[ind] = pred_a3
                # prediction_a4[ind] = pred_a4
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # prd_a1_itk = sitk.GetImageFromArray(prediction_a1.astype(np.float32))
        # prd_a2_itk = sitk.GetImageFromArray(prediction_a2.astype(np.float32))
        # prd_a3_itk = sitk.GetImageFromArray(prediction_a3.astype(np.float32))
        # prd_a4_itk = sitk.GetImageFromArray(prediction_a4.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        # prd_a1_itk.SetSpacing((1, 1, z_spacing))
        # prd_a2_itk.SetSpacing((1, 1, z_spacing))
        # prd_a3_itk.SetSpacing((1, 1, z_spacing))
        # prd_a4_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        # sitk.WriteImage(prd_a1_itk, test_save_path + '/' + case + "_pred_a1.nii.gz")
        # sitk.WriteImage(prd_a2_itk, test_save_path + '/' + case + "_pred_a2.nii.gz")
        # sitk.WriteImage(prd_a3_itk, test_save_path + '/' + case + "_pred_a3.nii.gz")
        # sitk.WriteImage(prd_a4_itk, test_save_path + '/' + case + "_pred_a4.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def train_acc(outputs, labels, classes):
    labels_numpy = labels.cpu().detach().numpy()  # [B, H, W]
    batch_size = outputs.shape[0]  # slice
    outputs_arg = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # [B, H, W]
    # outputs_arg = torch.argmax(torch.sigmoid(outputs), dim=1)  # [B, H, W]
    outputs_numpy = outputs_arg.cpu().detach().numpy()
    metric_list = []
    for j in range(1, classes):
        metric_list.append(calculate_metric_percase(outputs_numpy == j, labels_numpy == j))

    metric_i = np.array(metric_list)
    print(metric_i)
    # metric_i = metric_i / batch_size
    mean_dice = np.mean(metric_i, axis=0)[0]
    mean_hd = np.mean(metric_i, axis=0)[1]
    return mean_dice, mean_hd


