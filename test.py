import logging
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

import load_dataset
import RAP as fs
from dataset_specifics import get_label_names
from settings import Settings
from utils import Scores

SP_SLICES = 3
IMAGE_SIZE = 256
SHOTS = 1

logging.basicConfig(
    filename="./data/logs.log",
    format="[%(asctime)s] %(levelname)s %(name)s %(threadName)s : %(message)s",
)


def MR_normalize(x_in):
    return x_in / 255


def ts_main() -> None:
    settings = Settings()
    # _, data_params, net_params, train_params, eval_params = (
    #     settings["COMMON"],
    #     settings["DATA"],
    #     settings["NETWORK"],
    #     settings["TRAINING"],
    #     settings["EVAL"],
    # )

    net_params = settings["NETWORK"]

    # Deterministic setting for reproducibility.
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # initialize and load the trained model
    logging.info("Create model...")
    model = fs.RAP(net_params)
    model.load_state_dict(torch.load("./data/1000model.pth", map_location="cpu"))
    # model.cuda()
    model.eval()

    logging.info("Load data...")
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
        "supp_idx": 0,
    }
    n_part = 3

    test_dataset = load_dataset.TestDataset(data_config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Get unique labels (classes).
    labels = get_label_names("CHAOST2")

    # Loop over classes.
    class_dice = {}
    class_iou = {}

    logging.info("Starting validation...")
    for label_val, label_name in labels.items():
        # Skip BG class.
        if label_name == "BG":
            continue
        elif not np.intersect1d([label_val], data_config["test_label"]):
            continue

        logging.info(f"Test Class: {label_name}")

        # Get support sample + mask for the current class.
        support_sample = test_dataset.getSupport(
            label=label_val, all_slices=False, N=n_part
        )
        test_dataset.label = label_val

        with torch.no_grad():
            # Unpack support data.

            support_image = [
                support_sample["image"][[i]].float()
                for i in range(support_sample["image"].shape[0])
            ]  # 1 x n_part x 3 x H x W

            support_fg_mask = [
                support_sample["label"][[i]].float()
                for i in range(support_sample["image"].shape[0])
            ]  # 1 x n-part x H x W

            # s_input = torch.cat(support_image, 1)
            # s_mask = torch.cat(support_fg_mask, 1)
            # print(f"s_input shape: {s_input.shape}", f"s_mask shape: {s_mask.shape}")
            # assert False

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):
                # Unpack query data.
                query_image = [
                    sample["image"][i].float() for i in range(sample["image"].shape[0])
                ]  # [C x 3 x H x W]
                # assert False

                query_label = sample["label"].long()  # C x H x W
                query_id = sample["id"][0].split("image_")[1][: -len(".nii.gz")]

                # Compute output.
                # Match support slice and query sub-chunk.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample["image"].shape[1]
                idx_ = np.linspace(0, C_q, n_part + 1).astype("int")

                for sub_chunk in range(n_part):
                    support_image_s = support_image[sub_chunk]  # 1 x 3 x H x W
                    support_fg_mask_s = support_fg_mask[sub_chunk]  # 1 x H x W
                    # print(support_image_s.shape)

                    support = torch.cat([support_image_s, support_fg_mask_s], 2)

                    support = support.permute(1, 0, 2, 3, 4)
                    s_mask = support_fg_mask_s.permute(1, 0, 2, 3, 4)

                    query_image_s = query_image[0][
                        idx_[sub_chunk] : idx_[sub_chunk + 1]
                    ]  # C' x 3 x H x W

                    query_pred_s = []
                    query_pred_s2 = []
                    for i in range(query_image_s.shape[0]):
                        # _pred_s, _ = model(
                        #     [support_image_s],
                        #     [support_fg_mask_s],
                        #     [],
                        #     n_iters=n_part,
                        # )  # C x 2 x H x W

                        out, max_corr, sp_img_prior, sp_mask_prior = model(
                            query_image_s[[i]],
                            support,
                            s_mask,
                        )

                        out = F.interpolate(
                            out, size=[256, 256], mode="bilinear", align_corners=True
                        )

                        sp_pred = F.interpolate(
                            sp_mask_prior,
                            size=[256, 256],
                            mode="bilinear",
                            align_corners=True,
                        )

                        output = (out > 0.5).squeeze(1)
                        # sp_pred = (sp_pred > 0.5).squeeze(1)

                        # pred_mask.append(output.cpu().numpy())
                        # sp_mask.append(sp_pred.squeeze(1).cpu().numpy())
                        query_pred_s.append(out)
                        query_pred_s2.append(output.cpu().numpy())

                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s2 = np.concatenate(query_pred_s2, 0)

                    query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                    query_pred[idx_[sub_chunk] : idx_[sub_chunk + 1]] = query_pred_s

                # Record scores.
                scores.record(query_pred, query_label)

                # Log.
                # _log.info(
                #     f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.'
                # )
                # _log.info(f"Dice score: {scores.patient_dice[-1].item()}")

                # Save predictions.
                file_name: str = os.path.join(
                    "./data/prediction",
                    f"prediction_{query_id}_{label_name}.nii.gz",
                )

                itk_pred = sitk.GetImageFromArray(query_pred_s2.astype(np.uint8))
                sitk.WriteImage(itk_pred, file_name, True)
                logging.info(f"{query_id} has been saved. ")

            # Log class-wise results
            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            logging.info(f"Test Class: {label_name}")
            logging.info(f"Mean class IoU: {class_iou[label_name]}")
            logging.info(f"Mean class Dice: {class_dice[label_name]}")

    logging.info("Final results...")
    logging.info(f"Mean IoU: {class_iou}")
    logging.info(f"Mean Dice: {class_dice}")
    logging.info("End of validation.")


if __name__ == "__main__":
    ts_main()
