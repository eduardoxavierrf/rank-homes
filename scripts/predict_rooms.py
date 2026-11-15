"""Display an image annotated with detections from a locally trained RF-DETR."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "artifacts/runs/rooms-expanded-small/checkpoint_best_total.pth"


def predict_image(image, weights, threshold, device):
    import supervision as sv
    import torch
    from rfdetr import RFDETR

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFDETR.from_checkpoint(str(weights.resolve()), device=device)
    detections = model.predict(image, threshold=threshold)
    labels = [
        f"{name} {confidence:.0%}"
        for name, confidence in zip(detections.data["class_name"], detections.confidence)
    ]
    annotated = sv.BoxAnnotator().annotate(scene=image.copy(), detections=detections)
    annotated = sv.LabelAnnotator(text_padding=4, smart_position=True).annotate(
        scene=annotated, detections=detections, labels=labels
    )
    return annotated, len(detections), device


def show_image(image):
    import tkinter as tk
    from PIL import ImageTk

    window = tk.Tk()
    try:
        window.title("RF-DETR detections - press any key to close")
        preview = image.copy()
        preview.thumbnail((window.winfo_screenwidth() - 100, window.winfo_screenheight() - 150))
        photo = ImageTk.PhotoImage(preview, master=window)
        tk.Label(window, image=photo).pack()
        window.resizable(False, False)
        window.bind("<Key>", lambda event: window.quit())
        window.protocol("WM_DELETE_WINDOW", window.quit)
        window.mainloop()
    finally:
        window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained checkpoint (.pth)")
    parser.add_argument("--output", type=Path, help="Save to this path instead of displaying the image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence, between 0 and 1 (default: 0.5)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device (default: auto)")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if not args.image.is_file():
        parser.error(f"Input image not found: {args.image}")
    if not args.weights.is_file():
        parser.error(f"Trained checkpoint not found: {args.weights}. Select one with --weights.")
    output = args.output
    if output and output.resolve() == args.image.resolve():
        parser.error("--output must differ from the input image.")
    if output and Image.registered_extensions().get(output.suffix.lower()) not in Image.SAVE:
        parser.error(f"Unsupported output image extension: {output.suffix}")
    try:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        parser.error(f"Could not read input image: {error}")

    annotated, count, device = predict_image(image, args.weights, args.threshold, args.device)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output)
        print(f"Saved {count} detections ({device}): {output.resolve()}")
    else:
        print(f"Showing {count} detections ({device}). Close the window or press any key to exit.")
        show_image(annotated)


if __name__ == "__main__":
    main()
