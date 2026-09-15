"""
Финальный скрипт инференса для PatchCore.
Правильные ключи: pred_scores, pred_labels, anomaly_maps, pred_boxes.
"""
import sys
from pathlib import Path

import torch

# Патч torch.load для PyTorch 2.6+
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from anomalib.models import Patchcore
from anomalib.data import PredictDataset
from anomalib.engine import Engine

# === 1. Настройки ===
MODEL_PATH = Path("results/Patchcore/MVTec/bottle/v3/weights/lightning/model.ckpt")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_SIZE = (224, 224)   # ⚠️ 224, а не 256 — так делает Anomalib

print("=" * 60)
print("Инференс PatchCore")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ОШИБКА: модель не найдена: {MODEL_PATH}")
    sys.exit(1)

# === 2. Модель ===
print("\n[1/4] Создание модели...")
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)

# === 3. Engine ===
print("[2/4] Создание Engine...")
engine = Engine(accelerator="cpu", devices=1)

# === 4. Тестовые картинки ===
print("[3/4] Тестовые картинки...")
test_images = [
    ("norm", Path("data/MVTecAD/bottle/test/good/000.png")),
    ("defect_large", Path("data/MVTecAD/bottle/test/broken_large/000.png")),
    ("defect_small", Path("data/MVTecAD/bottle/test/broken_small/000.png")),
    ("contamination", Path("data/MVTecAD/bottle/test/contamination/000.png")),
]

# === 5. Инференс ===
print("[4/4] Инференс...\n")
print("-" * 60)

results = []
for name, img_path in test_images:
    if not img_path.exists():
        print(f"  Пропускаю {name}: файл не найден")
        continue

    dataset = PredictDataset(path=img_path, image_size=IMAGE_SIZE)
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=MODEL_PATH)

    if not predictions:
        print(f"  {name}: нет предсказания")
        continue

    pred = predictions[0]

    # === Извлекаем поля (правильные имена!) ===
    score = float(pred["pred_scores"].item())          # anomaly score 0..1
    label = int(pred["pred_labels"].item())            # 0 = норма, 1 = дефект
    anomaly_map = pred["anomaly_maps"].squeeze().cpu().numpy()  # (224, 224)
    pred_boxes = pred["pred_boxes"][0].cpu().numpy() if pred["pred_boxes"][0].numel() > 0 else None
    box_scores = pred["box_scores"][0].cpu().numpy() if pred["box_scores"][0].numel() > 0 else None

    label_str = "DEFECT" if label == 1 else "NORMAL"
    print(f"  {name}: score={score:.4f}, label={label_str}")

    # === Визуализация: оригинал + heatmap + overlay с боксом ===
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    img_np = np.array(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Исходник
    axes[0].imshow(img_np)
    axes[0].set_title(f"{name}\nИсходник", fontsize=11)
    axes[0].axis("off")

    # 2. Heatmap
    axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title(f"Anomaly Map\nscore={score:.3f}", fontsize=11)
    axes[1].axis("off")

    # 3. Overlay + bounding box
    axes[2].imshow(img_np)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.4)

    if pred_boxes is not None and len(pred_boxes) > 0:
        box = pred_boxes[0]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[2].add_patch(rect)
        bs = box_scores[0] if box_scores is not None else 0
        axes[2].set_title(f"Overlay + box\n{label_str} ({bs:.2f})", fontsize=11)
    else:
        axes[2].set_title(f"Overlay\n{label_str}", fontsize=11)

    axes[2].axis("off")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{name}_result.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    → {out_path}")

    results.append({"name": name, "score": score, "label": label_str})

# === 6. Итог ===
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)
for r in results:
    print(f"  {r['name']:20s} score={r['score']:.4f}  →  {r['label']}")

print(f"\nВизуализации: {OUTPUT_DIR.resolve()}")
print("Готово!")