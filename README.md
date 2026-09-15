<div align="center">

# 🔍 Defect Detection

**Промышленная система машинного зрения для обнаружения дефектов на производстве**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Anomalib](https://img.shields.io/badge/Anomalib-1.1.0-1B9AAA?style=for-the-badge)](https://github.com/openvinotoolkit/anomalib)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

*Anomaly detection без примеров дефектов — обучение только на нормальных образцах*

</div>

---

## 📊 Результаты

<div align="center">

| Метрика | Значение | Описание |
|:-------:|:--------:|:---------|
| 🎯 **Image AUROC** | **1.0000** | Точность классификации (дефект/норма) |
| 📊 **Image F1 Score** | **0.9920** | Баланс precision/recall |
| 🗺️ **Pixel AUROC** | **0.9815** | Точность локализации дефекта |
| 🔬 **Pixel F1 Score** | **0.7301** | Точность сегментации |

**Достигнут уровень Research SOTA** (PatchCore, CVPR 2022) на датасете MVTec AD.

</div>

---

## 🎯 Что делает система

<table>
<tr>
<td width="50%">

### Вход
- 📷 Изображение детали (PNG/JPG)
- 🔄 Размер: любой (нормализуется до 224×224)
- 📡 Источник: файл, камера, REST API

</td>
<td width="50%">

### Выход
- ✅ **Вердикт**: `NORMAL` или `DEFECT`
- 📊 **Score**: уверенность (0.0 – 1.0)
- 🗺️ **Heatmap**: карта аномалий
- 🟩 **Bounding box**: рамка дефекта

</td>
</tr>
</table>

---

## 🖼️ Примеры результатов

### ✅ NORMAL — нормальная бутылка (score 0.28)

<img src="docs/screenshots/norm_result.png" alt="NORMAL result" width="100%">

*Слева — исходник, в центре — карта аномалий (слабое кольцо по краю), справа — overlay. Модель верно определила норму.*

---

### ❌ DEFECT (small) — малый скол (score 0.60)

<img src="docs/screenshots/defect_small_result.png" alt="Small defect" width="100%">

*Небольшой скол на боку бутылки. Модель локализовала дефект — красное пятно в нужном месте.*

---

### ❌ DEFECT (contamination) — загрязнение (score 0.71)

<img src="docs/screenshots/contamination_result.png" alt="Contamination defect" width="100%">

*Загрязнение в центре. Модель точно указала область — heatmap сконцентрирован на дефекте.*

---

### ❌ DEFECT (large) — большой скол (score 0.83)

<img src="docs/screenshots/defect_large_result.png" alt="Large defect" width="100%">

*Крупный скол на всю высоту. Модель уверенно классифицировала как DEFECT с высоким score.*

---

**Логика score:** чем серьёзнее дефект — тем выше `anomaly_score`:

| Категория | Score |
|:---------:|:-----:|
| ✅ NORMAL | 0.28 |
| ❌ DEFECT (small) | 0.60 |
| ❌ DEFECT (contamination) | 0.71 |
| ❌ DEFECT (large) | 0.83 |

---

## 🏗️ Архитектура

### High-Level обзор

```mermaid
graph TB
    subgraph Client["🌐 Клиенты"]
        Browser[🌐 Браузер<br/>Drag&Drop UI]
        Curl[💻 curl / Postman]
        App[📱 Внешние приложения]
    end

    subgraph Service["⚙️ Сервис (порт 8020)"]
        API[🚀 FastAPI]
        Detector[🔍 DefectDetector]
    end

    subgraph Model["🧠 Модель"]
        PatchCore[🎯 PatchCore<br/>WideResNet50]
        Memory[(💾 Memory Bank<br/>~300 патчей)]
    end

    subgraph Storage["💿 Хранилище"]
        CKPT[(📦 model.ckpt<br/>214 МБ)]
        Output[(🖼️ Визуализации)]
    end

    Browser -->|POST /predict| API
    Curl -->|POST /predict| API
    App -->|POST /predict| API

    API -->|image| Detector
    Detector -->|inference| PatchCore
    PatchCore -.->|k-NN search| Memory
    PatchCore -->|load weights| CKPT
    Detector -->|heatmap + box| API
    API -->|JSON + base64| Browser
    API -->|PNG| Output

    style Browser fill:#3b82f6,color:#fff
    style Curl fill:#3b82f6,color:#fff
    style App fill:#3b82f6,color:#fff
    style API fill:#10b981,color:#fff
    style Detector fill:#f59e0b,color:#fff
    style PatchCore fill:#ef4444,color:#fff
    style Memory fill:#8b5cf6,color:#fff
    style CKPT fill:#6b7280,color:#fff
    style Output fill:#6b7280,color:#fff
```

---

### Детально: ML Pipeline

#### 🔄 Фаза 1. Обучение (Offline)

```mermaid
sequenceDiagram
    autonumber
    participant D as 📁 MVTec AD
    participant L as 🔄 DataLoader
    participant W as 🧠 WideResNet50
    participant C as ✂️ Coreset Sampler
    participant M as 💾 Memory Bank

    Note over D,M: Только нормальные образцы (209 шт)

    D->>L: Загрузка изображений
    L->>L: Resize 900×900 → 224×224
    L->>L: Normalize (ImageNet stats)
    L->>W: Батч из 32 изображений
    W->>W: Feature Extraction<br/>(layer2 + layer3)
    W->>C: ~1M патчей (224×224×1024)
    C->>C: Greedy k-center<br/>sampling_ratio=0.1
    C->>M: ~300 «представительных» патчей
    M->>M: Сохранение в .ckpt<br/>(214 МБ)

    Note over M: Модель обучена за 1 эпоху<br/>Время: ~5 минут CPU
```

#### 🎯 Фаза 2. Инференс (Online)

```mermaid
sequenceDiagram
    autonumber
    participant C as 🌐 Клиент
    participant A as 🚀 FastAPI
    participant D as 🔍 Detector
    participant W as 🧠 WideResNet50
    participant M as 💾 Memory Bank
    participant R as 📊 Postprocess

    C->>A: POST /predict<br/>(картинка)
    A->>A: Валидация формата
    A->>D: predict(image)
    D->>W: Feature Extraction
    W->>D: Патчи тестового изображения
    D->>M: k-NN поиск (k=9)
    M->>D: Расстояния до ближайших патчей
    D->>R: Anomaly score + map
    R->>R: Threshold → label
    R->>R: Bounding box (connected components)
    R->>R: Heatmap overlay (matplotlib)
    R->>A: JSON + base64 PNG
    A->>C: Response

    Note over C,R: Время: 1-3 сек на CPU
```

---

### Модули

```mermaid
graph LR
    subgraph Entry["🎬 Точки входа"]
        Train[train_patchcore.py<br/>Обучение]
        Predict[predict.py<br/>Batch инференс]
        Api[api.py<br/>REST сервис]
    end

    subgraph Core["🧩 Ядро"]
        Det[detector.py<br/>DefectDetector class]
        Anomalib[Anomalib<br/>PatchCore]
    end

    subgraph Data["📁 Данные"]
        Raw[data/MVTecAD<br/>Исходники]
        Result[results/<br/>.ckpt + logs]
        Viz[output/<br/>PNG визуализации]
    end

    Train --> Anomalib
    Predict --> Det
    Api --> Det
    Det --> Anomalib
    Anomalib --> Raw
    Anomalib --> Result
    Det --> Viz

    style Train fill:#fbbf24,color:#000
    style Predict fill:#fbbf24,color:#000
    style Api fill:#10b981,color:#fff
    style Det fill:#3b82f6,color:#fff
    style Anomalib fill:#ef4444,color:#fff
    style Raw fill:#6b7280,color:#fff
    style Result fill:#6b7280,color:#fff
    style Viz fill:#6b7280,color:#fff
```

---

## 🔬 Как работает PatchCore

**PatchCore** — метод anomaly detection, который **не требует примеров дефектов** при обучении. Ключевая идея — «банк памяти» из нормальных патчей.

### 📚 Три фазы

<table>
<tr>
<td width="33%" valign="top">

### 1️⃣ Feature Extraction

Каждое изображение пропускается через **предобученный WideResNet50**. Из промежуточных слоёв (`layer2`, `layer3`) извлекаются **карты признаков** — абстрактные «отпечатки» локальных областей.

**Что получаем:** для каждой позиции на картинке — вектор из 1024 чисел.

</td>
<td width="33%" valign="top">

### 2️⃣ Coreset Sampling

Из ~1 000 000 патчей отбираем **10% самых представительных** через **Greedy k-center** — алгоритм, который максимизирует разнообразие.

**Результат:** ~300 патчей в «банке памяти». Меньше памяти, быстрее поиск.

</td>
<td width="33%" valign="top">

### 3️⃣ Anomaly Scoring

Для новой картинки: извлекаем патчи → ищем **k=9 ближайших** в банке (k-NN) → усредняем расстояния.

**Логика:** если патч «далеко» от всех нормальных — это аномалия. Score = усреднённое расстояние.

</td>
</tr>
</table>

### 🎨 Почему это работает

```mermaid
graph TB
    subgraph Normal["✅ Нормальный патч"]
        N1[Извлечён из нормы]
        N2[Похож на банк памяти]
        N3[Score: низкий]
        N1 --> N2 --> N3
    end

    subgraph Defect["❌ Дефектный патч"]
        D1[Извлечён из дефекта]
        D2[Не похож на банк]
        D3[Score: высокий]
        D1 --> D2 --> D3
    end

    style N3 fill:#4ade80
    style D3 fill:#f87171
```

**Ключевое преимущество:** модель **никогда не видела дефектов**. Она выучила «как выглядит норма» и находит всё, что не подходит под это описание. Это решает главную боль реального производства — **дефекты редки, собирать датасет невозможно**.

---

## 🛠️ Технологический стек

<table>
<tr>
<td valign="top" width="50%">

### 🧠 Machine Learning
- **Python 3.11**
- **PyTorch 2.x** (CPU)
- **Anomalib 1.1.0** — фреймворк anomaly detection
- **PatchCore** — алгоритм (CVPR 2022)
- **WideResNet50** — backbone (timm)
- **scikit-learn** — метрики и утилиты

### 🌐 Backend
- **FastAPI** — REST API
- **Uvicorn** — ASGI-сервер
- **Pydantic** — валидация данных
- **python-multipart** — загрузка файлов

</td>
<td valign="top" width="50%">

### 📊 Данные и визуализация
- **MVTec AD** — датасет (bottle)
- **NumPy** — массивы
- **Pillow** — изображения
- **OpenCV** — CV-операции
- **Matplotlib** — графики и heatmap
- **Pandas** — таблицы

### 🚀 DevOps
- **venv** — изоляция окружения
- **requirements.txt** — фиксация версий
- **Git** — контроль версий
- **Jupyter Lab** — EDA

</td>
</tr>
</table>

---

## 📁 Структура проекта

```
defect-detection/
│
├── 🌐 api.py                    # FastAPI сервис (порт 8020)
├── 🧩 detector.py               # Класс DefectDetector
├── 🎯 predict.py                # Batch-инференс с визуализацией
├── 🎓 train_patchcore.py        # Скрипт обучения
├── 📋 requirements.txt          # Зависимости
├── 📝 README.md
├── 🚫 .gitignore
│
├── 📓 notebooks/
│   └── 01_eda.ipynb             # Разведочный анализ данных
│
├── 📊 docs/
│   ├── eda_summary.csv          # Статистика по дефектам
│   ├── eda_counts.csv           # Количество изображений
│   └── screenshots/             # Визуализации результатов
│       ├── norm_result.png
│       ├── defect_large_result.png
│       ├── defect_small_result.png
│       └── contamination_result.png
│
├── 📁 data/                     # ← в .gitignore
│   └── MVTecAD/bottle/
│       ├── train/good/          # 209 нормальных (обучение)
│       ├── test/                # 83 тестовых
│       │   ├── good/            # 20 норм
│       │   ├── broken_large/    # 20 больших сколов
│       │   ├── broken_small/    # 22 малых скола
│       │   └── contamination/   # 21 загрязнение
│       └── ground_truth/        # Маски дефектов
│
├── 🎯 results/                  # ← в .gitignore
│   └── Patchcore/MVTec/bottle/
│       └── v3/weights/lightning/
│           └── model.ckpt       # 214 МБ, обученная модель
│
└── 🖼️ output/                   # ← в .gitignore
    ├── norm_result.png
    ├── defect_large_result.png
    ├── defect_small_result.png
    └── contamination_result.png
```

---

## 🚀 Быстрый старт

### 📦 Предварительные требования

- **Python 3.11** (не 3.12+ — несовместимость с Anomalib)
- **~10 ГБ свободного места**
- **Kaggle API** для скачивания датасета

### 1️⃣ Клонирование и окружение

```bash
git clone <repo-url>
cd defect-detection

# Виртуальное окружение
python -m venv venv311
venv311\Scripts\activate              # Windows
source venv311/bin/activate           # Linux/Mac

# Зависимости
pip install -r requirements.txt
```

### 2️⃣ Данные

**Вариант A — через Kaggle CLI:**

```bash
# Получить kaggle.json: https://www.kaggle.com/settings → API → Create New Token
# Положить в C:\Users\<User>\.kaggle\access_token

kaggle datasets download -d ipythonx/mvtec-ad -p ./data/MVTecAD --unzip

# Оставить только bottle, удалить остальное
cd data/MVTecAD
rmdir /s /q cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
cd ../..
```

**Вариант B — вручную:**

Скачать с [официального сайта MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) и распаковать `bottle/` в `data/MVTecAD/`.

### 3️⃣ Обучение

```bash
python train_patchcore.py
```

**Что произойдёт:**
- Модель извлечёт патчи из 209 нормальных бутылок
- Создаст «банк памяти» с 10% лучших патчей
- Протестирует на 83 тестовых изображениях
- Сохранит результат в `results/Patchcore/MVTec/bottle/vN/`

**Время:** ~5–7 минут на CPU.

### 4️⃣ Запуск сервиса

```bash
python api.py
```

**Сервис доступен:**

| URL | Описание |
|-----|----------|
| 🌐 http://localhost:8020 | Drag&Drop UI |
| 📚 http://localhost:8020/docs | Swagger UI |
| ❤️ http://localhost:8020/health | Healthcheck |

### 5️⃣ Тестирование

**Через браузер:**

1. Открыть http://localhost:8020
2. Перетащить картинку
3. Получить результат

**Через curl:**

```bash
curl -X POST http://localhost:8020/predict \
     -F "file=@data/MVTecAD/bottle/test/broken_large/000.png"
```

---

## 📡 API Reference

### `GET /health`

Проверка статуса сервиса.

**Response:**
```json
{
  "status": "ok",
  "model_path": "results/Patchcore/MVTec/bottle/v3/weights/lightning/model.ckpt",
  "model_exists": true
}
```

---

### `POST /predict`

Детекция дефектов на изображении.

**Request:**
```http
POST /predict
Content-Type: multipart/form-data

file: <binary image>
```

**Response:**
```json
{
  "anomaly_score": 0.8253,
  "label": "DEFECT",
  "box": [74.0, 59.0, 211.0, 223.0],
  "box_score": 0.8459,
  "overlay_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "filename": "bottle.png"
}
```

**Поля:**

| Поле | Тип | Описание |
|------|-----|----------|
| `anomaly_score` | float | Степень аномальности (0.0–1.0) |
| `label` | string | `NORMAL` или `DEFECT` |
| `box` | array\|null | Bounding box `[x1, y1, x2, y2]` |
| `box_score` | float\|null | Уверенность рамки |
| `overlay_base64` | string | PNG с overlay (base64) |

---

## 📊 Данные

**MVTec AD** — стандартный бенчмарк для anomaly detection от MVTec Software.

<table>
<tr>
<td valign="top" width="50%">

### Статистика (категория `bottle`)

| Split | Класс | Кол-во |
|-------|-------|-------:|
| Train | good | 209 |
| Test | good | 20 |
| Test | broken_large | 20 |
| Test | broken_small | 22 |
| Test | contamination | 21 |
| **Всего** | | **292** |

**Размер изображений:** 900×900 px

</td>
<td valign="top" width="50%">

### Размеры дефектов

| Тип | Mean | Min | Max |
|-----|-----:|----:|----:|
| broken_large | 11.7% | 4.1% | 27.4% |
| broken_small | 3.1% | 0.9% | 8.1% |
| contamination | 8.5% | 0.6% | 16.5% |

**Вывод:** дефекты занимают 1–27% площади — модель должна быть чувствительной к мелким деталям.

</td>
</tr>
</table>

---

## 🎓 Как воспроизвести

### Проверка метрик

После обучения метрики выводятся в консоль:

```
============================================================
РЕЗУЛЬТАТЫ
============================================================
  image_AUROC: 1.0000
  image_F1Score: 0.9920
  pixel_AUROC: 0.9815
  pixel_F1Score: 0.7301
```

### Проверка на своих картинках

1. Положить картинку в `data/`
2. Открыть `predict.py`
3. Изменить `test_images` на свой путь
4. Запустить `python predict.py`
5. Смотреть результат в `output/`

---

## 🔮 Roadmap

- [x] ✅ Базовый RAG — PatchCore на MVTec AD
- [x] ✅ FastAPI сервис
- [x] ✅ HTML UI
- [ ] 🔄 Docker-контейнеризация
- [ ] ⚡ ONNX-экспорт (ускорение 2–3×)
- [ ] 🎓 Active Learning (дообучение на «неуверенных»)
- [ ] 🎥 Real-time видео (RTSP)
- [ ] 📱 Telegram-бот для оператора
- [ ] 🧊 3D-детекция (Real3D-AD)
- [ ] 🎨 Multi-category MVTec (15 категорий)

---

## 🧪 Технические детали

### Гиперпараметры PatchCore

| Параметр | Значение | Почему |
|----------|---------:|--------|
| `backbone` | `wide_resnet50_2` | Стандарт из статьи |
| `layers` | `["layer2", "layer3"]` | Средний + глубокий уровни |
| `coreset_sampling_ratio` | 0.1 | Компромисс точность/скорость |
| `num_neighbors` | 9 | k-NN для scoring |
| `image_size` | (224, 224) | Native размер backbone |

### Формат `.ckpt`

Файл 214 МБ содержит:

- **Веса WideResNet50** (~100 МБ, 25M параметров)
- **Memory Bank** (~10 МБ, ~300 патчей × 1024 float)
- **Состояние модели** (настройки, метрики)

Загрузка через `torch.load(path, weights_only=False)` — для собственных доверенных чекпоинтов.

---

## 🛠️ Challenges & Solutions

В процессе разработки мы столкнулись с рядом проблем, которые пришлось решать. Ниже — самые значимые.

### 🔴 Проблема 1: Конфликт версий в Anomalib

**Симптом:** `ImportError: cannot import name '_ActionSubCommands' from 'jsonargparse._actions'`

**Причина:** Anomalib 1.1.0 (2024 год) требует старую версию `jsonargparse` (< 4.32). pip при установке поставил свежую (4.52+), где внутренний класс переименован.

**Решение:**
```bash
pip install "jsonargparse==4.28.0"
```

**Что было дальше:** аналогичные конфликты с `rich` (15.0 ломал прогресс-бары) и `matplotlib` (3.10 удалил `tostring_rgb`). Все версии зафиксированы в `requirements.txt`.

---

### 🔴 Проблема 2: Скачивание MVTec AD

**Симптом 1:** `HTTPError: HTTP Error 404` — стандартный downloader Anomalib больше не работает, ссылка на сервер MVTec устарела.

**Симптом 2:** `RuntimeError: CAS Client Error` — новая CDN Hugging Face (Xet) выдаёт ошибку при загрузке.

**Симптом 3:** оригинальный датасет `Voxel51/mvtec-ad` весит ~5 ГБ (все 15 категорий), а нужна только `bottle`.

**Решение:** нашли альтернативный репозиторий на Hugging Face — [`visualanom/mirage_mvtec_visa`](https://huggingface.co/datasets/visualanom/mirage_mvtec_visa), где:
- Категории разбиты **отдельно** — можно скачать только `bottle` (~50 МБ вместо 5 ГБ).
- Структура совместима с Anomalib (после перемещения в `data/MVTecAD/bottle/`).
- Датасет — MIRAGE-версия MVTec AD, обогащённая синтетическими аномалиями.

**Команда:**
```bash
hf download --repo-type dataset --include "mvtec/bottle/**" --local-dir ./data/MVTecAD visualanom/mirage_mvtec_visa
```

---

### 🔴 Проблема 3: Отсутствующие зависимости Anomalib

**Симптом:** последовательные `ModuleNotFoundError` — `lightning`, `imgaug`, `kornia`, `open_clip`.

**Причина:** Anomalib 1.1.0 не указывает часть зависимостей как обязательные, ожидая ручной установки через `anomalib install` (который тянет ~500 МБ мусора).

**Решение:** установили только необходимый минимум вручную:
```bash
pip install "lightning>=2.0,<2.6" imgaug kornia open_clip_torch
```

---

### 🔴 Проблема 4: Конфликт NumPy 2.x

**Симптом:** `AttributeError: np.sctypes was removed in the NumPy 2.0 release`

**Причина:** `imgaug` (для аугментации) использует удалённый в NumPy 2.0 атрибут `np.sctypes`. При этом свежий `opencv-python 5.x` требует **именно** NumPy 2.x.

**Решение:** откатили оба пакета до совместимых версий:
```bash
pip install "numpy<2" "opencv-python<4.11"
```

Обе версии работают и с NumPy 1.x, и с 2.x.

---

### 🔴 Проблема 5: Права на символические ссылки в Windows

**Симптом:** `OSError: [WinError 1314] Клиент не обладает требуемыми правами` при попытке Anomalib создать папку `latest` как symlink на `v0`.

**Причина:** в Windows обычный пользователь не может создавать симлинки — нужен Developer Mode или права администратора.

**Решение:**
1. Включили **Developer Mode** (Настройки → Система → Для разработчиков).
2. Перезагрузили ПК.
3. Anomalib создал `results/Patchcore/MVTec/bottle/latest → v3` без ошибок.

---

### 🔴 Проблема 6: PyTorch 2.6 и `weights_only`

**Симптом:** `_pickle.UnpicklingError: Weights only load failed` при загрузке `.ckpt`.

**Причина:** в PyTorch 2.6 изменился дефолт `torch.load(weights_only=True)`. Anomalib сохранил в чекпоинт не только тензоры, но и объекты `torchvision.transforms.v2.Compose`, `Resize` и другие.

**Решение:** добавили патч-обёртку в скрипт инференса:
```python
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load
```

Безопасно, потому что мы загружаем **свой собственный** чекпоинт.

---

### 🔴 Проблема 7: Несовпадение API Anomalib 1.1.0

**Симптом:** `AttributeError: 'dict' object has no attribute 'pred_score'`

**Причина:** разные минорные версии Anomalib возвращают разные структуры из `engine.predict()`. В 1.1.0 это **словарь** с ключами `pred_scores`, `pred_labels`, `anomaly_maps` (во множественном числе).

**Решение:** узнали точные ключи через отладочный вывод и использовали их:
```python
score = float(pred["pred_scores"].item())
label = int(pred["pred_labels"].item())
anomaly_map = pred["anomaly_maps"].squeeze().cpu().numpy()
```

---

### ✅ Что это дало

В итоге мы научились:
- **Читать traceback** — искать конкретную строку, где упало.
- **Фиксировать версии** — `requirements.txt` защищает от будущих конфликтов.
- **Искать альтернативы** — когда стандартный путь не работает, искать обходной.
- **Работать с Windows-спецификой** — символические ссылки, права доступа.
- **Документировать проблемы** — чтобы команда не наступала на те же грабли.

---

## 📝 Лицензия

MIT License — свободно используйте, модифицируйте и распространяйте.

---

## 👤 Автор

**Максим Нагайцев**

- 💼 LinkedIn: [maksim-nagaytsev-ab2311432](https://www.linkedin.com/in/maksim-nagaytsev-ab2311432)
- 🐙 GitHub: [@Vaks911](https://github.com/Vaks911)

---

<div align="center">

⭐ **Если проект был полезен — поставьте звезду!**

*Сделано с ❤️ и большим количеством ночного кофе*

</div>