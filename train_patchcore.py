"""
Обучение PatchCore на датасете MVTec AD (категория bottle).
Использует Python API Anomalib вместо CLI.
"""
from pathlib import Path
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# === 1. Настройки ===
DATA_ROOT = Path("./data/MVTecAD")
CATEGORY = "bottle"
IMAGE_SIZE = (256, 256)   # размер изображений для модели

print("=" * 60)
print("Обучение PatchCore на MVTec AD / bottle")
print("=" * 60)
print(f"Данные: {DATA_ROOT.resolve()}")
print(f"Категория: {CATEGORY}")
print(f"Размер изображения: {IMAGE_SIZE}")

# === 2. Датасет ===
print("\n[1/4] Загрузка датасета...")
datamodule = MVTec(
    root=DATA_ROOT,
    category=CATEGORY,
    train_batch_size=32,   # сколько картинок за раз подаём на обучение
    eval_batch_size=32,    # сколько картинок за раз подаём на тест
    num_workers=0,         # 0 для Windows — иначе могут быть проблемы с multiprocessing
    image_size=IMAGE_SIZE,
)
datamodule.setup()
print(f"  Train: {len(datamodule.train_dataloader().dataset)} изображений")
print(f"  Test:  {len(datamodule.test_dataloader().dataset)} изображений")

# === 3. Модель ===
print("\n[2/4] Создание модели PatchCore...")
model = Patchcore(
    backbone="wide_resnet50_2",   # предобученная нейросеть для извлечения признаков
    layers=["layer2", "layer3"],  # какие слои использовать (средний и глубокий)
    coreset_sampling_ratio=0.1,   # оставляем 10% самых представительных патчей
    num_neighbors=9,              # для k-NN: сравниваем с 9 ближайшими
)
print(f"  Backbone: wide_resnet50_2")
print(f"  Coreset sampling: 10%")

# === 4. Engine (двигатель) ===
print("\n[3/4] Создание Engine...")
engine = Engine(
    max_epochs=1,          # PatchCore учится за 1 эпоху
    accelerator="cpu",     # используем CPU (без GPU)
    devices=1,
    default_root_dir="./results",  # куда сохранять результаты
)

# === 5. Обучение ===
print("\n[4/4] Обучение...")
print("-" * 60)
engine.fit(model=model, datamodule=datamodule)

# === 6. Тестирование ===
print("\n" + "-" * 60)
print("Тестирование...")
print("-" * 60)
test_results = engine.test(model=model, datamodule=datamodule)

# === 7. Вывод метрик ===
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)
if test_results:
    for metrics in test_results:
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

print("\nГотово! Модель сохранена в ./results/")