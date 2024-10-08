#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""

import os
import random
import shutil
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import load_dataset
import losses
import RAP as fs
from settings import Settings
from utils import set_logger

SP_SLICES = 3
IMAGE_SIZE = 256
SHOTS = 5


def MR_normalize(x_in):
    return x_in / 255


def train():
    logger = set_logger("./data/logs.log")
    save_path = "./data/prediction"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    os.mkdir(save_path + "/snapshorts")

    # Deterministic setting for reproducibility.
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    # cudnn.enabled = True
    # cudnn.benchmark = True

    # d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device=0)
    torch.set_num_threads(1)

    print("Create model...")
    logger.info("Create model...")
    settings = Settings()  # parse .ini
    common_params, data_params, net_params, train_params, _ = (
        settings["COMMON"],
        settings["DATA"],
        settings["NETWORK"],
        settings["TRAINING"],
        settings["EVAL"],
    )
    model = fs.RAP(net_params)
    # model = model.cuda()
    model.train()

    print("Set optimizer...")
    optim: dict[str, Any] = {
        "lr": train_params["learning_rate"],
        "weight_decay": train_params["optim_weight_decay"],
        "betas": train_params["optim_betas"],
        "eps": train_params["optim_eps"],
    }
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), **optim)
    lr_milestones: list[int] = [(ii + 1) * 1000 for ii in range(1000 // 1000 - 1)]

    scheduler = MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=train_params["lr_scheduler_gamma"]
    )

    # my_weight = torch.FloatTensor([0.1, 1.0])
    # criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # spatial_branch = fs.U_Network(2, [16, 32, 32, 32], [32, 32, 32, 32, 8, 8])
    # spatial_transformer = fs.SpatialTransformer((256, 256))

    mse = losses.MSE()
    smooth = losses.Grad(loss_mult=2)
    dice = losses.Dice()

    alpha = 0.05
    beta = 0.01

    print("Load data...")
    data_config = {
        "dataset": "CHAOST2",
        "n_shot": 1,
        "n_way": 1,
        "n_query": 1,
        "n_sv": 0,
        "max_iter": 1000,
        "eval_fold": 0,
        "min_size": 200,
        "max_slices": 3,
        "test_label": [1, 4],
        "exclude_label": [1, 4],
        "use_gt": True,
    }

    train_dataset = load_dataset.TrainDataset(data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    n_epochs = train_params["num_epochs"]
    log_loss: dict[str, int] = {"total_loss": 0, "query_loss": 0}

    i_iter = 0
    print("Start training...")
    for sub_epoch in range(n_epochs):
        print(f'This is epoch "{sub_epoch}" of "{n_epochs}" epochs.')
        for _, sample in enumerate(train_loader):
            # Prepare episode data.
            support_images = [
                [shot.float() for shot in way] for way in sample["support_images"]
            ]
            # [1, 1, 3, 256, 256]

            support_fg_mask = [
                [shot.float() for shot in way] for way in sample["support_fg_labels"]
            ]

            s_input: torch.Tensor = torch.cat(
                support_images[0], 1
            )  # [1, 1, 3, 256, 256]
            s_mask: torch.Tensor = torch.cat(
                support_fg_mask[0], 1
            )  # [1, 1, 3, 256, 256]

            support: torch.Tensor = torch.cat([s_input, s_mask], 2)

            support = support.permute(1, 0, 2, 3, 4)
            s_mask = s_mask.permute(1, 0, 2, 3, 4)

            query_images = [
                query_image.float() for query_image in sample["query_images"]
            ]
            query_images = torch.cat(query_images, dim=1)  # [1, 3, 256, 256]

            query_labels: torch.Tensor = torch.cat(
                [query_label.long() for query_label in sample["query_labels"]], dim=1
            )  # [1, 256, 256]

            # Compute outputs and losses.
            out, max_corr, sp_img_prior, sp_mask_prior = model(
                query_images,
                support,
                s_mask,
            )

            out = F.interpolate(
                out, size=[256, 256], mode="bilinear", align_corners=True
            )
            output = (out > 0.5).squeeze(1).float()

            sp_pred = F.interpolate(
                sp_mask_prior,
                size=[256, 256],
                mode="bilinear",
                align_corners=True,
            )

            sp_pred = (sp_pred > 0.5).squeeze(1).float()

            # flow = spatial_branch.forward(support[0][:, :3], query_images)
            # spatial_prior_support: torch.Tensor = spatial_transformer.forward(
            #     support[0][:, 1:2, ...], flow
            # )

            # spatial_prior_mask: torch.Tensor = spatial_transformer.forward(
            #     s_mask[0][:, 1:2, ...], flow
            # )

            loss_sp: torch.Tensor = (
                mse.loss(query_labels, sp_pred)
                + (alpha * dice.loss(query_labels, sp_mask_prior))
                + (beta * smooth.loss(query_labels, sp_img_prior))
            )

            loss_seg: torch.Tensor = dice.loss(query_labels, output)

            loss: torch.Tensor = loss_sp + loss_seg

            # set the gradients to zero
            optimizer.zero_grad()

            # back propagate and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            log_loss["total_loss"] += loss.item()

            # Print loss and take snapshots.
            if (i_iter + 1) % 1000 == 0:
                total_loss = log_loss["total_loss"] / 100

                log_loss["total_loss"] = 0

                print(f"step {i_iter + 1}: total_loss: {total_loss},")

            if (i_iter + 1) % 1000 == 0:
                print("###### Taking snapshot ######")
                torch.save(
                    model.state_dict(),
                    f"./data/prediction/snapshorts/{i_iter + 1}model.pth",
                )

            i_iter += 1

        if (sub_epoch + 1) == n_epochs:
            print("###### Taking snapshot ######")
            torch.save(
                model.state_dict(),
                f"./data/prediction/snapshorts/{i_iter + 1}model.pth",
            )
    print("End of training.")
    return 1


if __name__ == "__main__":
    train()
