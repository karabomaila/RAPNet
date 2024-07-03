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

import load_dataset
import RAP as fs
from settings import Settings

images_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/image*"
label_images_path = (
    r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/label*"
)

support_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/image*"
query_path = r"/home/karabo/code/Few-shot/data/CHAOST2/niis/T2SPIR/normalized/label*"

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

    all_query_img_path: list[str] = glob(query_path + "/*.nii.gz")
    all_support_img_path: list[str] = glob(support_path + "/*.nii.gz")

    save_path = "./prediction_la_dice_1000"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

    # initialize and load the trained model
    model = fs.RAP(net_params)
    model.load_state_dict(torch.load("./data/1000model.pth", map_location="cpu"))
    # model.cuda()
    model.eval()

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

    test_dataset = load_dataset.TestDataset(data_config)
    test_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

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

            nrrd.write(
                f"{save_path}/{query_name}_pred.nrrd",
                pred.transpose(2, 1, 0).astype(np.uint8),
            )
            nrrd.write(
                f"{save_path}/{query_name}_sp.nrrd",
                sp.transpose(2, 1, 0).astype(np.uint8),
            )


ts = ts_main()
