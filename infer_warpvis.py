"""

Generate optical flow with one of the available models.

This script can display and save optical flow estimated by any of the available models. It accepts multiple types of inputs,
including: individual images, a folder of images, a video, or a webcam stream.

"""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""

Modifications by Taewoo Suh

Visualize backwarped img2
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ptlflow import get_model, get_model_reference
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.flow_utils import flow_to_rgb, flow_write, flow_read
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import get_list_of_available_models_list, tensor_dict_to_numpy


def bwarp(x, flo):
    '''
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py#L237
    '''
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    #vgrid = torch.autograd.Variable(grid) + flo
    vgrid = grid + flo  # because we wont be backpropagating

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    #mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = torch.ones(x.size()).to(x.device)    # no need for backprop
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1
    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    return output * mask


def get_gaussian_weights(x, y, x1, x2, y1, y2, z=1.0):
    # z 0.0 ~ 1.0
    w11 = z * torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = z * torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = z * torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = z * torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(img, shiftx, shifty, weight):
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """

    N, C, H, W = img.size()

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(img.device).long().repeat(N, C,1,W).view(-1)
    flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(img.device).long().repeat(N, C,H,1).view(-1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(img.device).long().repeat(1, C, H, W).view(-1)
    idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(img.device).long().repeat(N, 1, H, W).view(-1)
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = torch.masked_select(ids, mask).clone().to(img.device)

    # Note here! accmulate fla must be true for proper bp
    img_warp = torch.zeros([N * C * H * W, ]).to(img.device)
    img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = torch.zeros([N * C * H * W, ]).to(img.device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


def fwarp(img, flo):

    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy
        https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
        https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py#L237
    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.size()

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(img, x1, y1, w11)
    img12, o12 = sample_one(img, x1, y2, w12)
    img21, o21 = sample_one(img, x2, y1, w21)
    img22, o22 = sample_one(img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=get_list_of_available_models_list(),
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Path to the inputs. It can be in any of these formats: 1. list of paths of images; 2. path to a folder "
            + "containing images; 3. path to a video; 4. the index of a webcam."
        ),
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help=(
            "(Optional) Path to the flow groundtruth. The path must point to one file, and --input_path must be composed of paths to two images only."
        ),
    )
    parser.add_argument(
        "--write_outputs",
        action="store_true",
        help="If set, the model outputs are saved to disk.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/inference")),
        help="Path to a folder where the results will be saved.",
    )
    parser.add_argument(
        "--flow_format",
        type=str,
        default="flo",
        choices=["flo", "png"],
        help="The format to use when saving the estimated optical flow.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, the results are shown on the screen.",
    )
    parser.add_argument(
        "--auto_forward",
        action="store_true",
        help=(
            "Only relevant if used with --show. If set, consecutive results will be shown without stopping. "
            + "Otherwise, each result remain on the screen until the user press a button."
        ),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[0, 0],
        help="If larger than zero, resize the input image before forwarding.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help=("Multiply the input image by this scale factor before forwarding."),
    )
    parser.add_argument(
        "--max_show_side",
        type=int,
        default=1000,
        help=(
            "If max(height, width) of the output image is larger than this value, then the image is downscaled "
            "before showing it on the screen."
        ),
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    return parser


@torch.no_grad()
def infer(args: Namespace, model: BaseModel) -> None:
    """Perform the inference.

    Parameters
    ----------
    model : BaseModel
        The model to be used for inference.
    args : Namespace
        Arguments to configure the model and the inference.

    See Also
    --------
    ptlflow.models.base_model.base_model.BaseModel : The parent class of the available models.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if args.fp16:
            model = model.half()

    cap, img_paths, num_imgs, prev_img = init_input(args.input_path)
    flow_gt = None
    if args.gt_path is not None:
        assert num_imgs == 2
        flow_gt = flow_read(args.gt_path)

    if args.scale_factor is not None:
        io_adapter = IOAdapter(
            model,
            prev_img.shape[:2],
            target_scale_factor=args.scale_factor,
            cuda=torch.cuda.is_available(),
            fp16=args.fp16,
        )
    else:
        io_adapter = IOAdapter(
            model,
            prev_img.shape[:2],
            args.input_size,
            cuda=torch.cuda.is_available(),
            fp16=args.fp16,
        )

    prev_dir_name = None
    for i in tqdm(range(1, num_imgs)):
        img, img_dir_name, img_name, is_img_valid = _read_image(cap, img_paths, i)
        if prev_dir_name is None:
            prev_dir_name = img_dir_name

        if not is_img_valid:
            break

        if img_dir_name == prev_dir_name:
            inputs = io_adapter.prepare_inputs([prev_img, img])
            preds = model(inputs)

            preds["images"] = inputs["images"]
            preds = io_adapter.unscale(preds)
            preds_npy = tensor_dict_to_numpy(preds)

            if flow_gt is not None:
                flow_pred = preds_npy["flows"]
                valid = ~np.isnan(flow_gt[..., 0])

                sq_dist = np.power(flow_pred - flow_gt, 2).sum(2)
                epe = np.sqrt(sq_dist[valid])

                gt_sq_dist = np.power(flow_gt, 2).sum(2)
                gt_dist_valid = np.sqrt(gt_sq_dist[valid])
                outlier = (epe > 3) & (epe > 0.05 * gt_dist_valid)
                print(
                    f"EPE: {epe.mean():.03f}, Outlier: {100*outlier.mean():.03f}",
                )

            preds_npy["flows_viz"] = flow_to_rgb(preds_npy["flows"])[:, :, ::-1]
            
            # backwarp img (=img2) by "flows"
            img2 = torch.tensor(img)[None, :, :, :].permute(0, 3, 1, 2)
            flow_pred = preds["flows"][0]
            img2_backwarped = bwarp(img2.cpu().float(), flow_pred.cpu())
            preds_npy["img2_bwarp"] = img2_backwarped.permute(0, 2, 3, 1)[0].cpu().numpy()

            # forward-warp prev_img (=img1) by "flows" - now fixed
            img1 = torch.tensor(prev_img)[None, :, :, :].permute(0, 3, 1, 2)
            flow_pred = preds["flows"][0]
            img1_forwarped = fwarp(img1.cpu().float(), flow_pred.cpu())
            preds_npy["img1_fwarp_imgw"] = img1_forwarped[0].permute(0, 2, 3, 1)[0].cpu().numpy()   # just to see what this even is
            preds_npy["img1_fwarp_o"] = img1_forwarped[1].permute(0, 2, 3, 1)[0].cpu().numpy()   # just to see what this even is
            Ft1 = preds_npy["img1_fwarp_imgw"]
            norm1 = preds_npy["img1_fwarp_o"]
            preds_npy["img1_fwarp_holemask"] = norm1 > 0
            Ft1[norm1 > 0] = Ft1[norm1 > 0] / norm1[norm1 > 0]
            preds_npy["img1_fwarp"] = Ft1   # trying something... works out pretty well?

            # tiling visualization
            # | Image 1     | Image 2   |
            # | Bwarp I2    | Fwarp I1  |
            #row1 = np.concatenate((img1.permute(0, 2, 3, 1)[0].cpu().numpy(), img2.permute(0, 2, 3, 1)[0].cpu().numpy()), axis=1)
            row1 = np.concatenate((prev_img, img), axis=1)
            row2 = np.concatenate((preds_npy["img2_bwarp"], preds_npy["img1_fwarp"]), axis=1)
            tiled_image = np.concatenate((row1, row2), axis=0)
            preds_npy["tiledvis_4"] = tiled_image

            if preds_npy.get("flows_b") is not None:
                preds_npy["flows_b_viz"] = flow_to_rgb(preds_npy["flows_b"])[:, :, ::-1]
            
            
            # TODO backwarp prev_img by "flows_b"
            # TODO forward-warp img by "flows_b"

            if args.write_outputs:
                write_outputs(
                    preds_npy,
                    args.output_path,
                    img_name,
                    args.flow_format,
                    img_dir_name,
                )
            if args.show:
                img1 = prev_img
                img2 = img
                if min(args.input_size) > 0:
                    img1 = cv.resize(prev_img, args.input_size[::-1])
                    img2 = cv.resize(img, args.input_size[::-1])
                key = show_outputs(
                    img1, img2, preds_npy, args.auto_forward, args.max_show_side
                )
                if key == 27:
                    break
        prev_dir_name = img_dir_name
        prev_img = img


def init_input(
    input_path: Union[str, List[str]]
) -> Tuple[cv.VideoCapture, List[Path], int, np.ndarray]:
    """Initialize the required variable to start loading the inputs.

    This function will detect which type of input_path was given (list of images, folder of images, video, or webcam).
    Then it will establish its length and also get the first frame of the input.

    Parameters
    ----------
    input_path : str
        The path to the input(s).

    Returns
    -------
    tuple[cv.VideoCapture, List[Path], int, np.ndarray]
        The initialized variables
        - a cv.VideoCapture if the input is a video OR
        - a list of paths to the images otherwise,
        - the maximum number of images, and
        - the first image.
    """
    cap = None
    img_paths = None
    if len(input_path) > 1:
        # Assumes it is a list of images
        img_paths = [Path(p) for p in input_path]
    else:
        input_path = Path(input_path[0])
        if input_path.is_dir():
            # Assumes it is a folder of images
            img_paths = sorted([p for p in input_path.glob("**/*") if not p.is_dir()])
        else:
            # Assumes it is a video or webcam index
            try:
                inp = int(input_path)
            except ValueError:
                pass
            cap = cv.VideoCapture(inp)

    if img_paths is not None:
        num_imgs = len(img_paths)
    else:
        # cv.VideoCapture does not always know the correct number of frames,
        # so we just set it as a high value
        num_imgs = 9999999

    if cap is not None:
        prev_img = cap.read()[1]
    else:
        prev_img = cv.imread(str(img_paths[0]))

    return cap, img_paths, num_imgs, prev_img


def show_outputs(
    img1: np.ndarray,
    img2: np.ndarray,
    preds_npy: Dict[str, np.ndarray],
    auto_forward: bool,
    max_show_side: int,
) -> int:
    """Show the images on the screen.

    Parameters
    ----------
    img1 : np.ndarray
        First image for estimating the optical flow.
    img2 : np.ndarray
        Second image for estimating the optical flow.
    preds_npy : dict[str, np.ndarray]
        The model predictions converted to numpy format.
    auto_forward : bool
        If false, the user needs to press a key to move to the next image.
    max_show_side : int
        If max(height, width) of the image is larger than this value, then it is downscaled before showing.

    Returns
    -------
    int
        A value representing which key the user pressed.

    See Also
    --------
    ptlflow.utils.utils.tensor_dict_to_numpy : This function can generate preds_npy.
    """
    preds_npy["img1"] = img1
    preds_npy["img2"] = img2
    for k, v in preds_npy.items():
        if len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3:
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(
                    v, (int(scale_factor * v.shape[1]), int(scale_factor * v.shape[0]))
                )
            cv.imshow(k, v)

    if auto_forward:
        w = 1
    else:
        w = 0
    key = cv.waitKey(w)
    return key


def write_outputs(
    preds_npy: Dict[str, np.ndarray],
    output_dir: str,
    img_name: str,
    flow_format: str,
    img_dir_name: Optional[str] = None,
) -> None:
    """Show the images on the screen.

    Parameters
    ----------
    preds_npy : dict[str, np.ndarray]
        The model predictions converted to numpy format.
    output_dir : str
        The path to the root dir where the outputs will be saved.
    img_name : str
        The name to be used to save each image (without extension).
    flow_format : str
        The format (extension) of the flow file to be saved. It can one of {flo, png}.

    See Also
    --------
    ptlflow.utils.utils.tensor_dict_to_numpy : This function can generate preds_npy.
    """
    for k, v in preds_npy.items():
        out_dir = Path(output_dir) / k
        if img_dir_name is not None:
            out_dir /= img_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_name
        if k == "flows" or k == "flows_b":
            if flow_format[0] != ".":
                flow_format = "." + flow_format
            flow_write(out_path.with_suffix(flow_format), v)
            print(f"Saved flow at: {out_path}")
        elif len(v.shape) == 2 or (
            len(v.shape) == 3 and (v.shape[2] == 1 or v.shape[2] == 3)
        ):
            # TODO implement differentiating between image types and backwarps
            if v.max() <= 1:
                v = v * 255
            cv.imwrite(str(out_path.with_suffix(".png")), v.astype(np.uint8))
            print(f"Saved image at: {out_path}")


def _read_image(
    cap: cv.VideoCapture, img_paths: List[Union[str, Path]], i: int
) -> Tuple[np.ndarray, str, bool]:
    if cap is not None:
        is_img_valid, img = cap.read()
        img_dir_name = None
        img_name = "{:08d}".format(i)
    else:
        img = cv.imread(str(img_paths[i]))
        img_dir_name = None
        if len(img_paths[i].parent.name) > 0:
            img_dir_name = img_paths[i].parent.name
        img_name = img_paths[i - 1].stem
        is_img_valid = True
    return img, img_dir_name, img_name, is_img_valid


if __name__ == "__main__":
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] != "-h" and sys.argv[1] != "--help":
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    args = parser.parse_args()

    model_id = args.model
    if args.pretrained_ckpt is not None:
        model_id += f"_{Path(args.pretrained_ckpt).stem}"
    args.output_path = Path(args.output_path) / model_id

    model = get_model(sys.argv[1], args.pretrained_ckpt, args)

    infer(args, model)
