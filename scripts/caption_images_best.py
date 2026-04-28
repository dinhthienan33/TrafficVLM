import argparse
import json
import os
from typing import Dict, List

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BEST_MODEL_NAME = "Salesforce/blip-image-captioning-large"


def get_image_paths(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    image_paths: List[str] = []
    for name in sorted(os.listdir(folder)):
        _, ext = os.path.splitext(name)
        if ext.lower() in VALID_EXTENSIONS:
            image_paths.append(os.path.join(folder, name))
    return image_paths


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def resolve_split_paths(dataset_root: str) -> Dict[str, str]:
    return {
        "train": os.path.join(dataset_root, "train", "images"),
        "val": os.path.join(dataset_root, "val", "images"),
        # bdd100k test images are directly under dataset/test
        "test": os.path.join(dataset_root, "test"),
    }


def load_existing(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def save_json(path: str, data: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@torch.inference_mode()
def caption_batch(
    paths: List[str],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    max_length: int,
    num_beams: int,
) -> List[str]:
    images: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Downloading/loading checkpoint: {args.model_name}")

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    split_paths = resolve_split_paths(args.dataset_root)

    for split, image_dir in split_paths.items():
        image_paths = get_image_paths(image_dir)
        if args.max_images > 0:
            image_paths = image_paths[: args.max_images]

        if len(image_paths) == 0:
            print(f"[{split}] No images found in {image_dir}")
            continue

        out_path = os.path.join(args.output_dir, f"{split}_captions.json")
        part_path = os.path.join(args.output_dir, f"{split}_captions.partial.json")
        results = load_existing(out_path) if args.resume else {}

        pending = [p for p in image_paths if os.path.basename(p) not in results]
        print(f"[{split}] Pending {len(pending)}/{len(image_paths)} images")

        if len(pending) == 0:
            continue

        for step, batch_paths in enumerate(batched(pending, args.batch_size), start=1):
            captions = caption_batch(
                batch_paths,
                processor=processor,
                model=model,
                device=device,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
            for p, cap in zip(batch_paths, captions):
                results[os.path.basename(p)] = cap.strip()

            if args.log_every > 0 and step % args.log_every == 0:
                done = len(image_paths) - len(pending) + min(step * args.batch_size, len(pending))
                print(f"[{split}] step {step}: captioned {done}/{len(image_paths)}")
                save_json(part_path, results)

        save_json(out_path, results)
        if os.path.isfile(part_path):
            os.remove(part_path)
        print(f"[{split}] Saved {len(results)} captions to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption dataset images using a strong BLIP checkpoint.")
    parser.add_argument("--dataset-root", type=str, default="/teamspace/studios/this_studio/dataset")
    parser.add_argument("--output-dir", type=str, default="/teamspace/studios/this_studio/dataset/captions_best")
    parser.add_argument("--model-name", type=str, default=BEST_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
