import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Make repo root importable when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_cfg_defaults
from models import TrafficVLM, get_tokenizer

TIME_SEGMENT_PATTERN = re.compile(r"<time=(\d+)> <time=(\d+)> (.+?)(?=<time=|\Z)")


def parse_generated_segments(text: str) -> List[str]:
    return [m[2] for m in TIME_SEGMENT_PATTERN.findall(text)]


CHECKPOINT_FOLDERS: Dict[str, Tuple[str, int]] = {
    "high_fps_all": ("https://drive.google.com/drive/folders/1PJjl4rTvGP-PqBESYiPk4SfPZJWuA9W0?usp=drive_link", 25),
    "local_temp_all": ("https://drive.google.com/drive/folders/1pCuQxsSUx9vNizdsJ2HNKVrh0I9pj073?usp=drive_link", 30),
}


def list_video_files(video_root: str) -> List[str]:
    if not os.path.isdir(video_root):
        return []
    videos = [f for f in sorted(os.listdir(video_root)) if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))]
    return [os.path.join(video_root, v) for v in videos]


def uniform_subsample(feats: torch.Tensor, max_feats: int) -> torch.Tensor:
    if len(feats) <= max_feats:
        return feats
    indices = torch.linspace(0, len(feats) - 1, steps=max_feats).long()
    return torch.index_select(feats, 0, indices)


def extract_video_features(
    video_path: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    target_fps: float,
    max_feats: int,
    clip_batch_size: int,
) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = 30.0
    frame_step = max(int(round(source_fps / target_fps)), 1)

    selected_frames: List[Image.Image] = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected_frames.append(Image.fromarray(frame_rgb))
        frame_id += 1
    cap.release()

    if len(selected_frames) == 0:
        return None

    all_feats: List[torch.Tensor] = []
    for i in range(0, len(selected_frames), clip_batch_size):
        batch_imgs = selected_frames[i : i + clip_batch_size]
        inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            image_feats = clip_model.get_image_features(**inputs).float()
        all_feats.append(image_feats.cpu())

    feats = torch.cat(all_feats, dim=0)
    feats = uniform_subsample(feats, max_feats)
    return feats


def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"epoch_(\d+)\.th$", os.path.basename(path))
    if m is None:
        return -1
    return int(m.group(1))


def auto_download_checkpoint(alias: str, out_dir: str) -> str:
    if alias not in CHECKPOINT_FOLDERS:
        raise ValueError(f"Unsupported checkpoint alias: {alias}. Available: {list(CHECKPOINT_FOLDERS.keys())}")

    url, preferred_epoch = CHECKPOINT_FOLDERS[alias]
    alias_dir = os.path.join(out_dir, alias)
    os.makedirs(alias_dir, exist_ok=True)

    try:
        subprocess.run(["gdown", "--version"], check=True, capture_output=True, text=True)
    except Exception:
        subprocess.run(["python", "-m", "pip", "install", "gdown", "--quiet"], check=True)

    subprocess.run(["gdown", "--folder", url, "-O", alias_dir], check=True)

    candidates = [os.path.join(alias_dir, f) for f in os.listdir(alias_dir) if f.endswith(".th")]
    if len(candidates) == 0:
        raise RuntimeError(f"No .th checkpoint found in {alias_dir} after download")

    preferred_name = f"epoch_{preferred_epoch}.th"
    for c in candidates:
        if os.path.basename(c) == preferred_name:
            return c

    candidates.sort(key=parse_epoch_from_name, reverse=True)
    return candidates[0]


def load_trafficvlm_model(
    experiment: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[TrafficVLM, dict]:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(f"experiments/{experiment}.yml")
    cfg.defrost()
    cfg.SOLVER.LOAD_FROM_PATH = checkpoint_path
    cfg.SOLVER.LOAD_FROM_EPOCH = -1
    cfg.freeze()

    tokenizer = get_tokenizer(cfg.MODEL.T5_PATH, cfg.DATA.NUM_BINS)
    model = TrafficVLM(
        cfg.MODEL,
        tokenizer,
        cfg.DATA.NUM_BINS,
        cfg.DATA.MAX_FEATS,
        cfg.DATA.SUB_FEATURE is not None,
        is_eval=True,
    ).to(device)

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()

    infer_cfg = {
        "num_beams": cfg.SOLVER.TEST.NUM_BEAMS,
        "top_p": cfg.SOLVER.TEST.TOP_P,
        "repetition_penalty": cfg.SOLVER.TEST.REPETITION_PENALTY,
        "length_penalty": cfg.SOLVER.TEST.LENGTH_PENALTY,
        "temperature": cfg.SOLVER.TEST.TEMPERATURE,
        "max_feats": cfg.DATA.MAX_FEATS,
        "sample_fps": cfg.DATA.FPS,
        "max_output_tokens": cfg.DATA.MAX_OUTPUT_TOKENS,
        "use_local": cfg.MODEL.USE_LOCAL,
        "max_phases": cfg.MODEL.MAX_PHASES,
        "use_sub_feat": cfg.DATA.SUB_FEATURE is not None,
    }
    return model, infer_cfg


@torch.inference_mode()
def generate_caption(
    model: TrafficVLM,
    feats: torch.Tensor,
    tgt_type: str,
    max_output_tokens: int,
    num_beams: int,
    top_p: float,
    repetition_penalty: float,
    length_penalty: float,
    temperature: float,
    local_batch: Optional[List[List[Optional[torch.Tensor]]]] = None,
) -> str:
    out = model.generate(
        feats=feats,
        tgt_type=tgt_type,
        local_batch=local_batch,
        use_nucleus_sampling=num_beams == 0,
        num_beams=num_beams,
        max_length=max_output_tokens,
        min_length=1,
        top_p=top_p if num_beams == 0 else 1.0,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        num_captions=1,
        temperature=temperature,
    )
    return out[0]


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = auto_download_checkpoint(args.checkpoint_alias, args.checkpoint_dir)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Using checkpoint: {checkpoint_path}")

    model, infer_cfg = load_trafficvlm_model(args.experiment, checkpoint_path, device)
    if args.max_output_tokens > 0:
        infer_cfg["max_output_tokens"] = args.max_output_tokens

    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    videos = list_video_files(args.video_root)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if len(videos) == 0:
        raise RuntimeError(f"No videos found in {args.video_root}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results: Dict[str, dict] = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            results = loaded
            print(f"Resume mode: loaded {len(results)} existing entries from {args.output_json}")

    for idx, video_path in enumerate(videos, start=1):
        vid_name = os.path.basename(video_path)
        if vid_name in results:
            continue
        feats = extract_video_features(
            video_path=video_path,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            target_fps=args.target_fps if args.target_fps > 0 else infer_cfg["sample_fps"],
            max_feats=infer_cfg["max_feats"],
            clip_batch_size=args.clip_batch_size,
        )
        if feats is None or len(feats) == 0:
            print(f"[{idx}/{len(videos)}] Skipped (no features): {vid_name}")
            continue

        feat_batch = feats.unsqueeze(0).to(device)
        local_batch: Optional[List[List[Optional[torch.Tensor]]]] = None
        if infer_cfg["use_local"]:
            # Keep local branch contract without requiring external local features.
            local_batch = [[None for _ in range(infer_cfg["max_phases"])]]

        vehicle_raw = generate_caption(
            model,
            feat_batch,
            "vehicle",
            infer_cfg["max_output_tokens"],
            infer_cfg["num_beams"],
            infer_cfg["top_p"],
            infer_cfg["repetition_penalty"],
            infer_cfg["length_penalty"],
            infer_cfg["temperature"],
            local_batch=local_batch,
        )
        pedestrian_raw = generate_caption(
            model,
            feat_batch,
            "pedestrian",
            infer_cfg["max_output_tokens"],
            infer_cfg["num_beams"],
            infer_cfg["top_p"],
            infer_cfg["repetition_penalty"],
            infer_cfg["length_penalty"],
            infer_cfg["temperature"],
            local_batch=local_batch,
        )

        vehicle_segments = parse_generated_segments(vehicle_raw)
        pedestrian_segments = parse_generated_segments(pedestrian_raw)

        results[vid_name] = {
            "vehicle_raw": vehicle_raw,
            "pedestrian_raw": pedestrian_raw,
            "vehicle_segments": vehicle_segments,
            "pedestrian_segments": pedestrian_segments,
            "num_frames_used": int(len(feats)),
        }
        if idx % args.log_every == 0:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[{idx}/{len(videos)}] Saved partial output -> {args.output_json}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved {len(results)} video captions -> {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce TrafficVLM inference on arbitrary videos.")
    parser.add_argument("--video-root", type=str, default="/teamspace/studios/this_studio/dataset_videos/1/video")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/teamspace/studios/this_studio/dataset_videos/trafficvlm_results.json",
    )
    parser.add_argument("--experiment", type=str, default="high_fps_all")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-alias", type=str, default="high_fps_all")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/teamspace/studios/this_studio/TrafficVLM/checkpoints",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--target-fps", type=float, default=0.0, help="0 means use experiment fps")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=0, help="0 means use experiment default")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means process all videos")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
