from argparse import ArgumentParser
import os

import torch

import numpy as np
import random

import cv2
from torchvision.ops import box_convert
from gs3dgs.scene.dataset_readers import readColmapSceneInfo
from sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import load_model, load_image, predict

import yaml

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_mask2d_groundingdino(
    prompt_list: list[str],
    data_dir: str,
    image_path_list: list[str],
    sam2_checkpoint: str = "checkpoints/sam2.1/sam2.1_hiera_large.pt",
    grounding_dino_checkpoint: str = "checkpoints/GroundingDINO/groundingdino_swinb_cogcoor.pth",
    grounding_dino_cfg: str = "submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    ext: str = "png",
    first_image_name: str = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    combine_prompt = True
):

    def get_best_boxes(
        boxes_multi: torch.Tensor,
        logits_multi: torch.Tensor,
        phrases_multi: list[str],
    ):
        best_boxes = [None for _ in prompt_list]
        best_logits = [-1 for _ in prompt_list]
        for _idx, box in enumerate(boxes_multi):
            phrase = phrases_multi[_idx]
            if phrase not in prompts_dict:
                continue
            cls = prompts_dict[phrases_multi[_idx]]
            conf = logits_multi[_idx].item()
            if best_boxes[cls] is None or conf > best_logits[cls]:
                best_boxes[cls] = box
                best_logits[cls] = conf
        best_boxes = torch.stack(best_boxes)
        return best_boxes

    @torch.enable_grad()
    def _iter_one_frame():
        image_path = image_path_list[frame_idx]
        frame = cv2.imread(image_path)

        output_image_name = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".png"

        masked = np.zeros((height, width), dtype=bool)
        for object_id, mask_org in zip(object_ids, masks):
            object_id = int(object_id)
            mask = mask_org[0].cpu().numpy() > 0
            mask = mask.astype(float)
            mask = (cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0).astype(float)
            mask = (cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0).astype(bool)
            # mask = cv2.blur(mask_org, (5, 5))
            masked |= mask
            mask_org_bin = mask.astype(np.uint8) * 255

            try:
                output_rgba = cv2.merge([frame, mask_org_bin])
            except Exception as e:
                print(f"Image shape: {frame.shape}, Mask shape: {mask_org_bin.shape}")
                print(image_path)
                raise e
            cv2.imwrite(os.path.join(mask_with_image_rgba_dir, prompt_list[object_id], output_image_name), output_rgba)

        masked_bin = masked.astype(np.uint8) * 255
        masked_output_rgba = cv2.merge([frame, masked_bin])
        cv2.imwrite(os.path.join(mask_with_image_rgba_dir, "masked", output_image_name), masked_output_rgba)

    if not os.path.exists(sam2_checkpoint):
        raise RuntimeError("Please download the checkpoint sam2_hiera_large.pt to checkpoints folder")
    if not os.path.exists(grounding_dino_checkpoint):
        raise RuntimeError("Please download the checkpoint groundingdino_swinb_cogcoor.pth to checkpoints folder")

    if not os.path.exists(grounding_dino_cfg):
        raise RuntimeError("Please download the config file GroundingDINO_SwinB_cfg.py to groundingdino/config folder")

    with torch.no_grad():
        mask_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        grounding_dino = load_model(grounding_dino_cfg, grounding_dino_checkpoint)

    prompts_preprocessed = prompt_list[0]
    for prompt in prompt_list[1:]:
        prompts_preprocessed += " . "
        prompts_preprocessed += prompt

    image_path_list.sort()

    if first_image_name is None:
        first_image_id = 0
    else:
        try:
            first_image_name = first_image_name + "." + ext
            first_image_id = [i for i, image_path in enumerate(image_path_list) if os.path.basename(image_path) == first_image_name][0]
        except IndexError:
            raise ValueError(f"Image {first_image_name} not found in the camera list")

    first_image_path = image_path_list[first_image_id]

    prompts_dict = {prompt: i for i, prompt in enumerate(prompt_list)}

    with torch.no_grad():
        state = mask_predictor.init_state(image_path_list)
        image_np, image_for_model = load_image(first_image_path)

        height, width = image_np.shape[:2]

        if combine_prompt:
            _boxes, _logits, _phrases = predict(
                model=grounding_dino,
                image=image_for_model,
                caption=prompts_preprocessed,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                remove_combined=True,
            )
        else:
            _boxes = torch.Tensor([])
            _logits = torch.Tensor([])
            _phrases = []

            results = [
                predict(
                    model=grounding_dino,
                    image=image_for_model,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    remove_combined=True
                )
                for caption in prompt_list
            ]

            all_boxes, all_logits, all_phrases = zip(*results)
            _boxes = torch.cat(all_boxes)
            _logits = torch.cat(all_logits)
            _phrases = sum(all_phrases, [])
        print(f"Detected objects: {list(set(_phrases))}")
        print(f"Prompts: {prompt_list}")
        if len(prompt_list) > len(_phrases):
            raise ValueError("The number of prompts is less than the number of detected objects")
        elif len(prompt_list) != len(set(_phrases)):
            if set(prompt_list) <= set(_phrases):
                print("Detected objects are a subset of the prompts")
            else:
                raise ValueError("Detected objects do not cover all prompts")
        elif len(prompt_list) < len(_phrases):
            print("Detected objects for each prompt is not unique")
        _best_boxes = get_best_boxes(_boxes, _logits, _phrases)

        _boxes_cxcywh = _best_boxes * torch.tensor([width, height, width, height])
        _boxes_xyxy = box_convert(boxes=_boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # add new prompts and instantly get the output on the same frame
        for idx, xyxy in enumerate(_boxes_xyxy):
            _, object_ids, masks = mask_predictor.add_new_points_or_box(
                state,
                box=xyxy.tolist(),
                frame_idx=first_image_id,
                obj_id=idx,
            )

    mask_with_image_rgba_dir = os.path.join(data_dir, "masked_image_rgba")
    os.makedirs(mask_with_image_rgba_dir, exist_ok=True)

    for prompt in prompt_list:
        os.makedirs(os.path.join(mask_with_image_rgba_dir, prompt), exist_ok=True)

    # os.makedirs(f"{mask_with_image_dir}/masked", exist_ok=True)
    os.makedirs(os.path.join(mask_with_image_rgba_dir, "masked"), exist_ok=True)

    with torch.no_grad():
        for frame_idx, object_ids, masks in mask_predictor.propagate_in_video(
            state, start_frame_idx=first_image_id - 1, reverse=True
        ):
            _iter_one_frame()

        for frame_idx, object_ids, masks in mask_predictor.propagate_in_video(
            state, start_frame_idx=first_image_id, reverse=False
        ):
            _iter_one_frame()

def segmentation(source_path: str):

    description_path = os.path.join(source_path, "description.yml")
    if not os.path.exists(description_path):
        raise FileNotFoundError(f"Config file not found at {description_path}")

    with open(description_path, "r") as f:
        description = yaml.safe_load(f)

    prompts = description["prompts"]
    prompt_list = prompts.split(",")
    prompt_list = [prompt.strip() for prompt in prompt_list]
    ext = description["ext"]
    first_image_name = description["first_image_name"]
    detect_conf = ""
    if "detect_conf" in description:
        detect_conf = description["detect_conf"]
    else:
        box_threshold = description["box_threshold"]
        text_threshold = description["text_threshold"]
    
    if "combine_prompt" in description:
        combine_prompt = description["combine_prompt"]
    else:
        combine_prompt = True


    image_path_list = [
        cam_info.image_path
        for cam_info in readColmapSceneInfo(source_path, "images", False).train_cameras
    ]

    get_mask2d_groundingdino(
        prompt_list=prompt_list,
        data_dir=source_path,
        image_path_list=image_path_list,
        ext=ext,
        first_image_name=first_image_name,
        box_threshold=box_threshold if detect_conf == "" else detect_conf,
        text_threshold=text_threshold if detect_conf == "" else detect_conf,
        combine_prompt=combine_prompt
    )

def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to source data")
    args = parser.parse_args()

    segmentation(args.source_path)

if __name__ == "__main__":
    main()
