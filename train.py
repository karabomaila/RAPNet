#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""

import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import load_dataset
import losses
import RAP as fs
from settings import Settings

SP_SLICES = 3
IMAGE_SIZE = 256
SHOTS = 5


def MR_normalize(x_in):
    return x_in / 255


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

    # Deterministic setting for reproducibility.
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

    print("Create model...")
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

    print("Set optimizer...")
    optim: dict[str, float] = {
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    }
    lr_step_gamma = 0.95
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_step_gamma)
    optimizer = torch.optim.SGD(model.parameters(), **optim)
    lr_milestones: list[int] = [(ii + 1) * 1000 for ii in range(1000 // 1000 - 1)]

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_step_gamma)

    my_weight = torch.FloatTensor([0.1, 1.0])
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    spatial_branch = fs.U_Network(2, [16, 32, 32, 32], [32, 32, 32, 32, 8, 8])
    spatial_transformer = fs.SpatialTransformer((256, 256))

    mse = losses.MSE()
    dice = losses.Dice()

    alpha = 0.05
    beta = 0.01

    print("Load data...")
    data_config = {
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

    n_sub_epochs = 1000 // 1000  # number of times for reloading
    log_loss: dict[str, int] = {"total_loss": 0, "query_loss": 0, "align_loss": 0}

    i_iter = 0
    print("Start training...")
    for sub_epoch in range(n_sub_epochs):
        print(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        for _, sample in enumerate(train_loader):
            pred_mask = []
            tmp_sprior = []
            sp_mask = []

            # Prepare episode data.
            # print(sample)
            support_images = [
                [shot.float() for shot in way] for way in sample["support_images"]
            ]
            # print(
            #     len(support_images), len(support_images[0]), support_images[0][0].shape
            # )

            support_fg_mask = [
                [shot.float() for shot in way] for way in sample["support_fg_labels"]
            ]
            # print(len(support_fg_mask), len(support_fg_mask[0]))

            s_input: torch.Tensor = torch.cat(support_images[0], 1)
            s_mask: torch.Tensor = torch.cat(support_fg_mask[0], 1)
            # print(f"s_input shape: {s_input.shape}", f"s_mask shape: {s_mask.shape}")

            support: torch.Tensor = torch.cat([s_input, s_mask], 2)
            # print(support.shape)
            support = support.permute(1, 0, 2, 3, 4)
            s_mask = s_mask.permute(1, 0, 2, 3, 4)

            query_images = [
                query_image.float() for query_image in sample["query_images"]
            ]
            query_images = torch.cat(query_images, dim=1)

            query_labels: torch.Tensor = torch.cat(
                [query_label.long() for query_label in sample["query_labels"]], dim=1
            )

            # Compute outputs and losses.
            # print(query_image.shape, support_image.shape, support_mask.shape)
            out, sp_pred, max_corr2, sp_img_prior = model(
                query_images,
                support,
                s_mask,
            )

            tmp_sprior.append(out.detach().cpu().numpy())
            # print(out.shape, query_images.shape)

            out = F.interpolate(
                out, size=[256, 256], mode="bilinear", align_corners=True
            )

            sp_pred = F.interpolate(
                sp_pred,
                size=[256, 256],
                mode="bilinear",
                align_corners=True,
            )

            output = (out > 0.5).squeeze(1)
            sp_pred = (sp_pred > 0.5).squeeze(1)

            pred_mask.append(output.cpu().numpy())
            sp_mask.append(sp_pred.squeeze(1).cpu().numpy())

            # pred = np.concatenate(pred_mask, 0)
            # sp = np.concatenate(sp_mask, 0)

            flow = spatial_branch.forward(support[0][:, :3], query_images)
            spatial_prior_support: torch.Tensor = spatial_transformer.forward(
                support[0][:, 1:2, ...], flow
            )

            spatial_prior_mask: torch.Tensor = spatial_transformer.forward(
                s_mask[0][:, 1:2, ...], flow
            )

            loss_sp: torch.Tensor = mse.loss(
                query_labels, spatial_prior_support
            ) + alpha * dice.loss(query_labels, spatial_prior_mask)

            #  y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            loss_seg: torch.Tensor = dice.loss(query_labels, output)

            loss: torch.Tensor = loss_sp + loss_seg

            # query_loss = criterion(
            #     torch.log(
            #         torch.clamp(
            #             output.cpu().numpy(),
            #             torch.finfo(torch.float32).eps,
            #             1 - torch.finfo(torch.float32).eps,
            #         )
            #     ),
            #     query_labels,
            # )
            # support_loss = criterion(
            #     torch.log(
            #         torch.clamp(
            #             sp_pred.cpu().numpy(),
            #             torch.finfo(torch.float32).eps,
            #             1 - torch.finfo(torch.float32).eps,
            #         )
            #     ),
            #     support,
            # )

            # loss = query_loss + support_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            # calculate total loss
            # weights = [0.5, 0.5]
            # loss = 0
            # loss_list = []
            # for n, loss_function in enumerate(losses):
            #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            #     loss_list.append(curr_loss.item())
            #     loss += curr_loss

            # back propagate and optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            # query_loss = query_loss.detach().data.cpu().numpy()
            # align_loss = align_loss.detach().data.cpu().numpy()

            log_loss["total_loss"] += loss.item()
            # log_loss["query_loss"] += query_loss
            # log_loss["align_loss"] += align_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % 100 == 0:
                total_loss = log_loss["total_loss"] / 100
                # query_loss = log_loss["query_loss"] / 100
                # align_loss = log_loss["align_loss"] / 100

                log_loss["total_loss"] = 0
                # log_loss["query_loss"] = 0
                # log_loss["align_loss"] = 0

                print(f"step {i_iter + 1}: total_loss: {total_loss}," f" align_loss: 0")

            if (i_iter + 1) % 1000 == 0:
                print("###### Taking snapshot ######")
                torch.save(
                    model.state_dict(),
                    f"./data/{i_iter + 1}model.pth",
                )

            i_iter += 1
    print("End of training.")
    return 1


if __name__ == "__main__":
    train()
