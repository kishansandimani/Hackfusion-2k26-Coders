"""Measure how close ChipPal's counts are to counts you made by hand.

1. Put test photos in a folder called testset/
2. Make testset/truth.csv with one row per photo (count each part type yourself):
       image,ic,capacitor,connector,resistor
       board1.jpg,4,31,3,45
3. Run:  python evaluate.py        (then python evaluate.py --enhance to compare)
Class names in the header must match your model's class names (upper or lower case).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

FOLDER = Path("testset")
TILE = 640


def count_accuracy(pred, truth):
    """1.0 means every count matched. Each miss or extra detection lowers it."""
    total = sum(truth.values())
    err = sum(abs(pred.get(c, 0) - n) for c, n in truth.items())
    return max(0.0, 1 - err / total) if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--enhance", action="store_true", help="clean up dark or blurry photos first")
    args = ap.parse_args()
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights,
        confidence_threshold=args.conf, device="cpu")
    rows = list(csv.DictReader(open(FOLDER / "truth.csv", encoding="utf-8")))
    classes = [c.strip().lower() for c in rows[0] if c != "image"]
    sum_true, sum_pred, accs = Counter(), Counter(), []
    for r in rows:
        truth = {c: int(r.get(c) or r.get(c.upper()) or 0) for c in classes}
        path = str(FOLDER / r["image"])
        if args.enhance:
            from PIL import Image
            from enhance import enhance_if_needed
            fixed, _ = enhance_if_needed(Image.open(path).convert("RGB"))
            path = "_eval_tmp.jpg"
            fixed.save(path)
        res = get_sliced_prediction(
            path, model, slice_height=TILE, slice_width=TILE,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        found = Counter(p.category.name.lower() for p in res.object_prediction_list)
        pred = {c: found.get(c, 0) for c in classes}
        acc = count_accuracy(pred, truth)
        accs.append(acc)
        sum_true.update(truth)
        sum_pred.update(pred)
        print(f"{r['image']:<24} count accuracy {acc:6.1%}   " +
              "  ".join(f"{c} {pred[c]}/{truth[c]}" for c in classes))
    print("\nPer class (predicted / true over all photos):")
    for c in classes:
        t, p = sum_true[c], sum_pred[c]
        print(f"  {c:<18} {p:>5} / {t:<5}  count accuracy {count_accuracy({c: p}, {c: t}):6.1%}")
    print(f"\nAverage count accuracy over {len(accs)} photos: {sum(accs) / len(accs):.1%}")
    print("Note: this compares counts only. It does not check that each box is on the right part.")


if __name__ == "__main__":
    main()