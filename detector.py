"""
Класс DefectDetector — обёртка над моделью PatchCore.
Загружает модель один раз, потом делает предсказания.
"""
import torch
from pathlib import Path

# Патч torch.load для PyTorch 2.6+
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from PIL import Image
import numpy as np

from anomalib.models import Patchcore
from anomalib.data import PredictDataset
from anomalib.engine import Engine


class DefectDetector:
    """Обёртка над моделью PatchCore для инференса."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Загружает модель из .ckpt.

        Args:
            model_path: путь к файлу .ckpt
            device: "cpu" или "cuda"
        """
        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")

        print(f"Загрузка модели из {self.model_path}...")

        # Создаём модель с теми же параметрами, что при обучении
        self.model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )

        # Engine
        self.engine = Engine(accelerator=device, devices=1)

        # Прогреваем — делаем один фиктивный проход,
        # чтобы модель загрузила веса в память
        print("Модель создана. Готова к инференсу.")

    def predict(self, image_path: str) -> dict:
        """
        Делает предсказание для одной картинки.

        Args:
            image_path: путь к PNG/JPG картинке

        Returns:
            dict с полями:
                - anomaly_score (float): 0..1
                - label (str): "NORMAL" или "DEFECT"
                - pred_label_int (int): 0 или 1
                - box (list | None): [x1, y1, x2, y2] или None
                - box_score (float | None): уверенность рамки
                - anomaly_map (np.ndarray): 2D карта аномалий
                - image_size (tuple): размер препроцессинга
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Картинка не найдена: {img_path}")

        # Датасет из одной картинки
        dataset = PredictDataset(path=img_path, image_size=(224, 224))

        # Предсказание
        predictions = self.engine.predict(
            model=self.model,
            dataset=dataset,
            ckpt_path=self.model_path,
        )

        if not predictions:
            raise RuntimeError("Модель не вернула предсказание")

        pred = predictions[0]

        # Извлекаем поля
        score = float(pred["pred_scores"].item())
        label_int = int(pred["pred_labels"].item())
        label_str = "DEFECT" if label_int == 1 else "NORMAL"
        anomaly_map = pred["anomaly_maps"].squeeze().cpu().numpy()

        # Bounding box (может отсутствовать)
        box = None
        box_score = None
        if pred["pred_boxes"][0].numel() > 0:
            box = pred["pred_boxes"][0][0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            if pred["box_scores"][0].numel() > 0:
                box_score = float(pred["box_scores"][0][0].item())

        return {
            "anomaly_score": score,
            "label": label_str,
            "pred_label_int": label_int,
            "box": box,
            "box_score": box_score,
            "anomaly_map": anomaly_map,
            "image_size": (224, 224),
        }