#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""


import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from utils import *
from RAP import RAP
import copy
import os
import random
import shutil
from glob import glob
import cv2
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import RAP as fs
from settings import Settings
import load_dataset 

images_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/image*"
label_images_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/label*"

support_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/image*"
query_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/label*"

SP_SLICES = 3
IMAGE_SIZE = 256
SHOTS = 5

def MR_normalize(x_in):
    return x_in / 255


def ts_main(ckpt_path):
    print("running?")
    settings = Settings()  # parse .ini
    common_params, data_params, net_params, train_params, eval_params = (
        settings["COMMON"],
        settings["DATA"],
        settings["NETWORK"],
        settings["TRAINING"],
        settings["EVAL"],
    )

    # initialize and load the trained model
    model = fs.RAP(net_params)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.cuda()
    model.eval()


    all_images_paths = glob(images_path + "/*.nii.gz")
    all_label_images_paths = glob(label_images_path + "/*.nii.gz")

    all_images_paths = sorted(all_images_paths, key=lambda x: int(x.split('_')[-1].split(".nii.gz")[0]))
    all_label_images_paths = sorted(all_label_images_paths, key=lambda x: int(x.split('_')[-1].split(".nii.gz")[0]))

    all_query_img_path = glob(query_path + "/*.nii.gz")
    all_support_img_path = glob(support_path + "/*.nii.gz")

    save_path = "./prediction_la_dice_1000"
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # else:
    #     shutil.rmtree(save_path)
    #     os.mkdir(save_path)

    # data flow and pred
    support_images = {}
    support_images_labels = {}
    query_images = {}
    query_images_labels = {}
    with torch.no_grad():
        for pid in all_query_img_path:
            print("qid:", pid)
            query_name = pid.split("\\")[-1].split(".")[0]

            img_query = nrrd.read(pid)[0].transpose(2, 1, 0)
            mask_query = nrrd.read(pid.replace("im", "m"))[0].transpose(2, 1, 0)

            tmp_support_path = copy.deepcopy(all_support_img_path)
            try:
                tmp_support_path.remove(pid)
            except:
                pass

            pred_mask = []
            tmp_sprior = []
            sp_mask = []
            for query_slice in range(img_query.shape[0]):
                if SP_SLICES == 1:

                    input = cv2.resize(
                        img_query[query_slice],
                        dsize=(IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    input = MR_normalize(input)
                    # 3 or 1 channel input
                    # input = torch.from_numpy(np.repeat(input[np.newaxis, np.newaxis, ...], 3, 1)).float().cuda()
                    query = (
                        torch.from_numpy(input[np.newaxis, np.newaxis, ...])
                        .float()
                        .cuda()
                    )

                else:
                    # sp_slices == 3
                    input = cv2.resize(
                        img_query[query_slice],
                        dsize=(IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    input = MR_normalize(input)
                    query = torch.from_numpy(input[np.newaxis, np.newaxis, ...]).float()
                    
                    if query_slice == 0:
                        query_pre = query
                    else:
                        input = cv2.resize(
                            img_query[query_slice - 1],
                            dsize=(IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        input = MR_normalize(input)
                        query_pre = torch.from_numpy(
                            input[np.newaxis, np.newaxis, ...]
                        ).float()

                    if query_slice == img_query.shape[0] - 1:
                        query_next = query
                    else:
                        input = cv2.resize(
                            img_query[query_slice + 1],
                            dsize=(IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        input = MR_normalize(input)
                        query_next = torch.from_numpy(
                            input[np.newaxis, np.newaxis, ...]
                        ).float()

                    # finish read query img(1 or 3 slices) and mask (1 slice)
                    query = torch.cat([query_pre, query, query_next], dim=1).cuda()
                    mask_query = cv2.resize(
                        mask_query[query_slice],
                        dsize=(IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # every slice a support
                support_paths = random.sample(tmp_support_path, SHOTS)
                print("sids:", support_paths)

                sp_imgs = []
                sp_msks = []
                for i in range(SHOTS):
                    img_support = nrrd.read(support_paths[i])[0].transpose(2, 1, 0)
                    mask_support = (
                        nrrd.read(support_paths[i].replace("_im", "_m"))[0]
                        .transpose(2, 1, 0)
                        .astype(np.uint8)
                    )
                    sp_imgs.append(img_support)
                    sp_msks.append(mask_support)

                # get cur_slice support
                s_inputs = []
                s_masks = []
                cond_inputs = []
                for i in range(SHOTS):
                    img_support = sp_imgs[i]
                    mask_support = sp_msks[i]
                    sp_shp0 = img_support.shape[0]

                    if SP_SLICES == 1:
                        sp_index = int(
                            query_slice / img_query.shape[0] * img_support.shape[0]
                        )

                        img_support = cv2.resize(
                            img_support[sp_index],
                            dsize=(IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        img_support = MR_normalize(img_support)
                        s_input = (
                            torch.from_numpy(
                                img_support[np.newaxis, np.newaxis, np.newaxis, ...]
                            )
                            .float()
                            .cuda()
                        )

                        msk_support = cv2.resize(
                            mask_support[sp_index],
                            dsize=(IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        s_mask = (
                            torch.from_numpy(
                                msk_support[np.newaxis, np.newaxis, np.newaxis, ...]
                            )
                            .float()
                            .cuda()
                        )
                    else:
                        # S1
                        # sp_index = sp_shp0//2

                        # S2
                        # bias = sp_shp0 / 3 / 2
                        # ratio = query_slice / img_query.shape[0]
                        # if ratio < 1 / 3:
                        #     sp_index = int(bias)
                        # elif ratio >= 1 / 3 and ratio < 2 / 3:
                        #     sp_index = int(1 / 3 * sp_shp0 + bias)
                        # else:
                        #     sp_index = int(2 / 3 * sp_shp0 + bias)

                        # S3
                        sp_index = int(query_slice / img_query.shape[0] * sp_shp0)

                        sp_indexes = [
                            max(sp_index - 1, 0),
                            sp_index,
                            min(sp_index + 1, sp_shp0 - 1),
                        ]
                        sp_imgs_tmp = []
                        sp_masks_tmp = []
                        for sp_index in sp_indexes:
                            img_support_r = cv2.resize(
                                img_support[sp_index],
                                dsize=(IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            img_support_r = MR_normalize(img_support_r)
                            s_input = (
                                torch.from_numpy(
                                    img_support_r[
                                        np.newaxis, np.newaxis, np.newaxis, ...
                                    ]
                                )
                                .float()
                                .cuda()
                            )
                            sp_imgs_tmp.append(s_input)
                            msk_support = (
                                cv2.resize(
                                    mask_support[sp_index],
                                    dsize=(IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                == 1
                            )
                            s_mask = (
                                torch.from_numpy(
                                    msk_support[np.newaxis, np.newaxis, np.newaxis, ...]
                                )
                                .float()
                                .cuda()
                            )
                            sp_masks_tmp.append(s_mask)

                        s_input = torch.cat(sp_imgs_tmp, 2)  # [1,1,slice,H,W]
                        s_mask = torch.cat(sp_masks_tmp, 2)  # [1,1,slice,H,W]

                    s_inputs.append(s_input)
                    s_masks.append(s_mask)

                # finish read support img and mask
                s_input = torch.cat(s_inputs, 1)  # 1, Kshot, slice, h, w
                s_mask = torch.cat(s_masks, 1)

                # # run model
                support = torch.cat([s_input, s_mask], 2)  # b, Kshot, slice*2, h, w
                cond_inputs_ = support.permute(1, 0, 2, 3, 4)  # Kshot, b, slice*2, h, w

                # forward
                out, sp_pred, max_corr2 = model.segmentor(
                    query, cond_inputs_, s_mask.permute(1, 0, 2, 3, 4)
                )

                tmp_sprior.append(out.detach().cpu().numpy())
                out = F.interpolate(
                    out, size=img_query.shape[1:], mode="bilinear", align_corners=True
                )
                sp_pred = F.interpolate(
                    sp_pred,
                    size=img_query.shape[1:],
                    mode="bilinear",
                    align_corners=True,
                )

                output = (out > 0.5).squeeze(1)
                sp_pred = (sp_pred > 0.5).squeeze(1)

                pred_mask.append(output.cpu().numpy())
                sp_mask.append(sp_pred.squeeze(1).cpu().numpy())

            pred = np.concatenate(pred_mask, 0)
            sp = np.concatenate(sp_mask, 0)

            nrrd.write(
                f"{save_path}/{query_name}_pred.nrrd",
                pred.transpose(2, 1, 0).astype(np.uint8),
            )
            nrrd.write(
                f"{save_path}/{query_name}_sp.nrrd",
                sp.transpose(2, 1, 0).astype(np.uint8),
            )


# print("running?")
# ckpt_path = "/home/karabo/code/Few-shot/data/rap.pth"
# ts = ts_main(ckpt_path)





def train():
    # if _run.observers:
        # Set up source folder
        # os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        # for source_file, _ in _run.experiment_info['sources']:
        #     os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
        #                 exist_ok=True)
        #     _run.observers[0].save_file(source_file, f'source/{source_file}')
        # shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # # Set up logger -> log to .txt
        # file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        # file_handler.setLevel('INFO')
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        # file_handler.setFormatter(formatter)
        # _log.handlers.append(file_handler)
        # _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    # if _config['seed'] is not None:
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    print(torch.cuda.is_available())
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # torch.cuda.set_device(device=0)
    torch.set_num_threads(1)

    print('Create model...')
    settings = Settings()  # parse .ini
    common_params, data_params, net_params, train_params, eval_params = (
        settings["COMMON"],
        settings["DATA"],
        settings["NETWORK"],
        settings["TRAINING"],
        settings["EVAL"],
    )
    model = fs.RAP(net_params)
    # model = model.cuda()
    model.train()

    print(f'Set optimizer...')
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }
    lr_step_gamma = 0.95
    optimizer = torch.optim.SGD(model.parameters(), **optim)
    lr_milestones = [(ii + 1) * 1000 for ii in
                     range(1000 // 1000 - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_step_gamma)

    my_weight = torch.FloatTensor([0.1, 1.0])
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    print(f'Load data...')
    data_config = {
            'n_shot': 1,
            'n_way': 1,
            'n_query': 1,
            'n_sv': 0,
            'max_iter': 1000,
            'eval_fold': 0,
            'min_size': 200,
            'max_slices': 3,
            'test_label': [1, 4],
            'exclude_label': None,
            'use_gt': True,
        }
    train_dataset = load_dataset.TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True)

    n_sub_epochs = 1000 // 1000  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0}

    i_iter = 0
    print(f'Start training...')
    for sub_epoch in range(n_sub_epochs):
        print(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        for _, sample in enumerate(train_loader):
            # Prepare episode data.
            print(sample)
            support_images = [[shot.float() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float() for shot in way]
                               for way in sample['support_fg_labels']]

            query_images = [query_image.float() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long() for query_label in sample['query_labels']], dim=0)

            # Compute outputs and losses.
            query_pred, align_loss = model(query_images, support_images, support_fg_mask)

            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)
            loss = query_loss + align_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy()

            # _run.log_scalar('total_loss', loss.item())
            # _run.log_scalar('query_loss', query_loss)
            # _run.log_scalar('align_loss', align_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # Print loss and take snapshots.
            if (i_iter + 1) %  100 == 0:
                total_loss = log_loss['total_loss'] / 100
                query_loss = log_loss['query_loss'] / 100
                align_loss = log_loss['align_loss'] / 100

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['align_loss'] = 0

                print(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss},'
                          f' align_loss: {align_loss}')

            if (i_iter + 1) % 1000 == 0:
                print('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           f"/home/karabo/code/Few-shot/data/{i_iter + 1}model.pth"
                           )

            i_iter += 1
    print('End of training.')
    return 1

if __name__ == "__main__":
    train()