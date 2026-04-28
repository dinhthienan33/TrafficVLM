import argparse
import json
import os
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_image_paths(split_path: str) -> List[str]:
    if not os.path.isdir(split_path):
        return []
    files = []
    for name in sorted(os.listdir(split_path)):
        _, ext = os.path.splitext(name)
        if ext.lower() in VALID_EXTENSIONS:
            files.append(os.path.join(split_path, name))
    return files


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@torch.inference_mode()
def caption_batch(
    image_paths: List[str],
    processor: ViTImageProcessor,
    tokenizer: AutoTokenizer,
    model: VisionEncoderDecoderModel,
    device: torch.device,
    max_length: int,
    num_beams: int,
) -> List[str]:
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams,
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def resolve_split_paths(dataset_root: str) -> Dict[str, str]:
    return {
        "train": os.path.join(dataset_root, "train", "images"),
        "val": os.path.join(dataset_root, "val", "images"),
        # bdd100k test images are directly under dataset/test
        "test": os.path.join(dataset_root, "test"),
    }


def load_existing_captions(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_json(path: str, data: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    split_paths = resolve_split_paths(args.dataset_root)

    for split, image_dir in split_paths.items():
        image_paths = get_image_paths(image_dir)
        if args.max_images > 0:
            image_paths = image_paths[: args.max_images]

        if len(image_paths) == 0:
            print(f"[{split}] No images found in: {image_dir}")
            continue

        out_path = os.path.join(args.output_dir, f"{split}_captions.json")
        partial_path = os.path.join(args.output_dir, f"{split}_captions.partial.json")
        result: Dict[str, str] = load_existing_captions(out_path) if args.resume else {}
        if args.resume and len(result) > 0:
            print(f"[{split}] Resume enabled: found {len(result)} existing captions in {out_path}")

        pending = [p for p in image_paths if os.path.basename(p) not in result]
        print(f"[{split}] Processing {len(pending)} pending images from {image_dir} (total {len(image_paths)})")
        if len(pending) == 0:
            continue

        for step, batch_paths in enumerate(batched(pending, args.batch_size), start=1):
            captions = caption_batch(
                batch_paths,
                processor=processor,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )

            for p, cap in zip(batch_paths, captions):
                result[os.path.basename(p)] = cap.strip()

            if args.log_every > 0 and step % args.log_every == 0:
                done = len(image_paths) - len(pending) + min(step * args.batch_size, len(pending))
                print(f"[{split}] step {step}: captioned {done}/{len(image_paths)}")
                save_json(partial_path, result)

        save_json(out_path, result)
        if os.path.isfile(partial_path):
            os.remove(partial_path)
        print(f"[{split}] Saved {len(result)} captions -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption all images in BDD100K-style dataset folders.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/teamspace/studios/this_studio/dataset",
        help="Root folder containing train/val/test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/teamspace/studios/this_studio/dataset/captions",
        help="Output folder for JSON caption files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nlpconnect/vit-gpt2-image-captioning",
        help="HF image captioning model name",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 means process all images in each split")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON if present")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
