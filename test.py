import copy
import logging
import os
import random

import cv2
import nrrd
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

    all_query_img_path: list[str] = []
    all_support_img_path: list[str] = []

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

        # Test.
        with torch.no_grad():
            model.eval()

            # Unpack support data.
            support_image = [
                support_sample["image"][[i]].float()
                for i in range(support_sample["image"].shape[0])
            ]  # n_shot x 3 x H x W
            support_fg_mask = [
                support_sample["label"][[i]].float()
                for i in range(support_sample["image"].shape[0])
            ]  # n_shot x H x W

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):
                # Unpack query data.
                query_image = [
                    sample["image"][i].float() for i in range(sample["image"].shape[0])
                ]  # [C x 3 x H x W]
                query_label = sample["label"].long()  # C x H x W
                query_id = sample["id"][0].split("image_")[1][: -len(".nii.gz")]

                # Compute output.
                # Match support slice and query sub-chunck.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample["image"].shape[1]
                idx_ = np.linspace(0, C_q, n_part + 1).astype("int")

                for sub_chunck in range(n_part):
                    support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                    query_image_s = query_image[0][
                        idx_[sub_chunck] : idx_[sub_chunck + 1]
                    ]  # C' x 3 x H x W

                    query_pred_s = []
                    for i in range(query_image_s.shape[0]):
                        _pred_s, _ = model(
                            [support_image_s],
                            [support_fg_mask_s],
                            [query_image_s[[i]]],
                            n_iters=n_part,
                        )  # C x 2 x H x W
                        query_pred_s.append(_pred_s)

                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                    query_pred[idx_[sub_chunck] : idx_[sub_chunck + 1]] = query_pred_s

                # Record scores.
                scores.record(query_pred, query_label)

                # Log.
                # _log.info(
                #     f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.'
                # )
                # _log.info(f"Dice score: {scores.patient_dice[-1].item()}")

                # Save predictions.
                file_name: str = os.path.join(
                    "./data",
                    f"prediction_{query_id}_{label_name}.nii.gz",
                )
                itk_pred = sitk.GetImageFromArray(query_pred)
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

    # data flow and pred
    with torch.no_grad():
        for pid in all_query_img_path:
            print("qid:", pid)
            query_name: str = pid.split("\\")[-1].split(".")[0]

            # read in the query image and its mask
            img_query = nrrd.read(pid)[0].transpose(2, 1, 0)
            mask_query = nrrd.read(pid.replace("im", "m"))[0].transpose(2, 1, 0)

            tmp_support_path: list[str] = copy.deepcopy(all_support_img_path)
            try:
                tmp_support_path.remove(pid)
            except Exception:
                pass

            pred_mask = []
            tmp_sprior = []
            sp_mask = []

            for query_slice in range(img_query.shape[0]):
                input = cv2.resize(
                    img_query[query_slice],
                    dsize=(IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )

                # preprocess the query slice
                input = MR_normalize(input)
                # 3 or 1 channel input
                # input = torch.from_numpy(np.repeat(input[np.newaxis, np.newaxis, ...], 3, 1)).float().cuda()
                query = torch.from_numpy(input[np.newaxis, np.newaxis, ...]).float()

                if SP_SLICES == 3:
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

                    # finish reading query img(1 or 3 slices) and mask (1 slice)
                    # combine the slices
                    query: torch.Tensor = torch.cat(
                        [query_pre, query, query_next], dim=1
                    ).cuda()

                    mask_query = cv2.resize(
                        mask_query[query_slice],
                        dsize=(IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # every slice a support
                support_paths: list[str] = random.sample(tmp_support_path, SHOTS)
                print("sids:", support_paths)

                # read in the sampled K-shot support images
                sp_imgs = []
                sp_masks = []
                for i in range(SHOTS):
                    img_support = nrrd.read(support_paths[i])[0].transpose(2, 1, 0)
                    mask_support = (
                        nrrd.read(support_paths[i].replace("_im", "_m"))[0]
                        .transpose(2, 1, 0)
                        .astype(np.uint8)
                    )

                    sp_imgs.append(img_support)
                    sp_masks.append(mask_support)

                # get the current slice support images
                s_inputs = []
                s_masks = []
                for i in range(SHOTS):
                    img_support = sp_imgs[i]
                    mask_support = sp_masks[i]
                    sp_shp0: int = img_support.shape[0]

                    sp_index = int(
                        query_slice / img_query.shape[0] * img_support.shape[0]
                    )

                    if SP_SLICES == 1:
                        img_support = cv2.resize(
                            img_support[sp_index],
                            dsize=(IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        img_support = MR_normalize(img_support)
                        s_input = torch.from_numpy(
                            img_support[np.newaxis, np.newaxis, np.newaxis, ...]
                        ).float()

                        msk_support = cv2.resize(
                            mask_support[sp_index],
                            dsize=(IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        s_mask = torch.from_numpy(
                            msk_support[np.newaxis, np.newaxis, np.newaxis, ...]
                        ).float()
                    else:
                        sp_indexes: list[int] = [
                            max(sp_index - 1, 0),
                            sp_index,
                            min(sp_index + 1, sp_shp0 - 1),
                        ]

                        # reading the previous, current and the next support slices
                        sp_imgs_tmp = []
                        sp_masks_tmp = []
                        for sp_index in sp_indexes:
                            img_support_r = cv2.resize(
                                img_support[sp_index],
                                dsize=(IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            img_support_r = MR_normalize(img_support_r)

                            s_input = torch.from_numpy(
                                img_support_r[np.newaxis, np.newaxis, np.newaxis, ...]
                            ).float()

                            msk_support = (
                                cv2.resize(
                                    mask_support[sp_index],
                                    dsize=(IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                == 1
                            )

                            s_mask = torch.from_numpy(
                                msk_support[np.newaxis, np.newaxis, np.newaxis, ...]
                            ).float()

                            sp_imgs_tmp.append(s_input)
                            sp_masks_tmp.append(s_mask)

                        # add or combine all the slices
                        s_input: torch.Tensor = torch.cat(
                            sp_imgs_tmp, 2
                        )  # [1,1,slice,H,W]
                        s_mask: torch.Tensor = torch.cat(
                            sp_masks_tmp, 2
                        )  # [1,1,slice,H,W]

                    s_inputs.append(s_input)
                    s_masks.append(s_mask)

                # finish read support img and mask
                s_input = torch.cat(s_inputs, 1)  # 1, Kshot, slice, h, w
                s_mask = torch.cat(s_masks, 1)

                # run model
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

            # nrrd.write(
            #     f"{save_path}/{query_name}_pred.nrrd",
            #     pred.transpose(2, 1, 0).astype(np.uint8),
            # )
            # nrrd.write(
            #     f"{save_path}/{query_name}_sp.nrrd",
            #     sp.transpose(2, 1, 0).astype(np.uint8),
            # )


if __name__ == "__main__":
    ts_main()
