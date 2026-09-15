"""
FastAPI-сервис для детекции дефектов.
Порт: 8020
"""
import io
import base64
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI, для сервера
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from detector import DefectDetector

# === 1. Настройки ===
MODEL_PATH = "results/Patchcore/MVTec/bottle/v3/weights/lightning/model.ckpt"
PORT = 8020

# === 2. Приложение ===
app = FastAPI(
    title="Defect Detection API",
    description="Детекция дефектов на бутылках (PatchCore + MVTec AD)",
    version="1.0.0",
)

# === 3. Загружаем модель ОДИН РАЗ ===
print("=" * 60)
print("Запуск сервиса Defect Detection")
print("=" * 60)
detector = DefectDetector(model_path=MODEL_PATH, device="cpu")
print("Сервис готов!")


# === 4. Эндпоинты ===

@app.get("/", response_class=HTMLResponse)
def index():
    """HTML-страница для загрузки картинок."""
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Defect Detection</title>
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 1100px;
                margin: 40px auto;
                padding: 20px;
                background: #0f172a;
                color: #e2e8f0;
                line-height: 1.5;
            }
            h1 { color: #38bdf8; margin-bottom: 8px; }
            .subtitle { color: #94a3b8; margin-bottom: 30px; font-size: 15px; }

            .dropzone {
                display: block;
                border: 2px dashed #38bdf8;
                border-radius: 12px;
                padding: 60px 20px;
                text-align: center;
                background: #1e293b;
                margin-bottom: 20px;
                cursor: pointer;
                transition: 0.2s;
            }
            .dropzone:hover { background: #334155; border-color: #60a5fa; }

            #fileInput { display: none; }

            .dropzone-icon { font-size: 48px; margin-bottom: 12px; }
            .dropzone-text { font-size: 17px; font-weight: 500; color: #e2e8f0; }
            .dropzone-hint { font-size: 13px; color: #94a3b8; margin-top: 6px; }

            #loading { display: none; text-align: center; padding: 20px; color: #38bdf8; font-size: 16px; }

            #result { display: none; margin-top: 30px; }
            #result.show { display: block; }

            .metrics { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
            .metric {
                padding: 12px 20px;
                border-radius: 8px;
                background: #1e293b;
                border: 1px solid #334155;
            }
            .metric-label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
            .metric-value { font-size: 18px; font-weight: 600; color: #38bdf8; }

            .label-NORMAL { color: #4ade80; }
            .label-DEFECT { color: #f87171; }

            .overlay-img { max-width: 100%; border-radius: 8px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>🔍 Defect Detection</h1>
        <p class="subtitle">Загрузите изображение бутылки — модель найдёт дефекты (царапины, сколы, загрязнения)</p>

        <input type="file" id="fileInput" accept="image/*">
        <label for="fileInput" class="dropzone">
            <div class="dropzone-icon">📤</div>
            <div class="dropzone-text">Нажмите, чтобы выбрать файл</div>
            <div class="dropzone-hint">PNG, JPG</div>
        </label>

        <div id="loading">⏳ Обработка... (несколько секунд)</div>

        <div id="result">
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Score</div>
                    <div class="metric-value" id="score">—</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Label</div>
                    <div class="metric-value" id="label">—</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Box</div>
                    <div class="metric-value" id="box">—</div>
                </div>
            </div>
            <img id="overlay" class="overlay-img" alt="overlay">
        </div>

        <script>
            const input = document.getElementById("fileInput");
            const loading = document.getElementById("loading");
            const result = document.getElementById("result");

            input.addEventListener("change", async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                loading.style.display = "block";
                result.classList.remove("show");

                const formData = new FormData();
                formData.append("file", file);

                try {
                    const resp = await fetch("/predict", {
                        method: "POST",
                        body: formData,
                    });
                    const data = await resp.json();

                    if (!resp.ok) throw new Error(data.detail || "Ошибка");

                    document.getElementById("score").textContent = data.anomaly_score;
                    const labelEl = document.getElementById("label");
                    labelEl.textContent = data.label;
                    labelEl.className = "metric-value label-" + data.label;

                    document.getElementById("box").textContent = data.box
                        ? data.box.map(v => Math.round(v)).join(", ")
                        : "нет";

                    document.getElementById("overlay").src = "data:image/png;base64," + data.overlay_base64;
                    result.classList.add("show");
                } catch (err) {
                    alert("Ошибка: " + err.message);
                } finally {
                    loading.style.display = "none";
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает картинку, возвращает результат детекции.
    
    Возвращает JSON:
    - anomaly_score: 0..1
    - label: NORMAL / DEFECT
    - box: [x1, y1, x2, y2] или None
    - box_score: уверенность рамки
    - overlay_base64: PNG с визуализацией (base64)
    """
    # Проверка типа файла
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Ожидается изображение, получено: {file.content_type}",
        )

    # Читаем байты
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть картинку: {e}")

    # Сохраняем во временный файл
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"upload_{file.filename}"
    image.save(tmp_path)

    # Инференс
    try:
        result = detector.predict(str(tmp_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    # Строим визуализацию
    img_resized = image.resize(result["image_size"])
    img_np = np.array(img_resized)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Исходник", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(result["anomaly_map"], cmap="jet")
    axes[1].set_title(f"Anomaly Map\nscore={result['anomaly_score']:.3f}", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(result["anomaly_map"], cmap="jet", alpha=0.4)

    if result["box"] is not None:
        x1, y1, x2, y2 = result["box"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[2].add_patch(rect)

    axes[2].set_title(f"Overlay\n{result['label']}", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()

    # Сохраняем в base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Возвращаем JSON
    return {
        "anomaly_score": round(result["anomaly_score"], 4),
        "label": result["label"],
        "box": result["box"],
        "box_score": round(result["box_score"], 4) if result["box_score"] is not None else None,
        "overlay_base64": overlay_b64,
        "filename": file.filename,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """HTML-страница для загрузки картинок."""
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Defect Detection</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                max-width: 1100px;
                margin: 40px auto;
                padding: 20px;
                background: #0f172a;
                color: #e2e8f0;
            }
            h1 { color: #38bdf8; }
            .dropzone {
                border: 2px dashed #38bdf8;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                background: #1e293b;
                margin-bottom: 20px;
                cursor: pointer;
                transition: 0.2s;
            }
            .dropzone:hover { background: #334155; }
            .dropzone input { display: none; }
            #result {
                margin-top: 30px;
                display: none;
            }
            #result.show { display: block; }
            .metric {
                display: inline-block;
                padding: 10px 20px;
                border-radius: 8px;
                background: #1e293b;
                margin-right: 15px;
                margin-bottom: 15px;
            }
            .metric strong { color: #38bdf8; }
            .label-normal { color: #4ade80; }
            .label-defect { color: #f87171; }
            img { max-width: 100%; border-radius: 8px; }
            .hint { color: #94a3b8; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>🔍 Defect Detection</h1>
        <p class="hint">Загрузите изображение бутылки — модель найдёт дефекты (царапины, сколы, загрязнения).</p>

        <label for="fileInput" class="dropzone">
            <input type="file" id="fileInput" accept="image/*">
            <div>
                <div style="font-size: 40px;">📤</div>
                <div>Нажмите, чтобы выбрать файл</div>
                <div class="hint">PNG, JPG</div>
            </div>
        </label>

        <div id="loading" style="display: none;">⏳ Обработка...</div>

        <div id="result">
            <div class="metric">Score: <strong id="score">—</strong></div>
            <div class="metric">Label: <strong id="label">—</strong></div>
            <div class="metric">Box: <strong id="box">—</strong></div>
            <img id="overlay" alt="overlay">
        </div>

        <script>
            const input = document.getElementById("fileInput");
            const loading = document.getElementById("loading");
            const result = document.getElementById("result");

            input.addEventListener("change", async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                loading.style.display = "block";
                result.classList.remove("show");

                const formData = new FormData();
                formData.append("file", file);

                try {
                    const resp = await fetch("/predict", {
                        method: "POST",
                        body: formData,
                    });
                    const data = await resp.json();

                    if (!resp.ok) throw new Error(data.detail || "Ошибка");

                    document.getElementById("score").textContent = data.anomaly_score;
                    const labelEl = document.getElementById("label");
                    labelEl.textContent = data.label;
                    labelEl.className = data.label === "DEFECT" ? "label-defect" : "label-normal";

                    document.getElementById("box").textContent = data.box
                        ? data.box.map(v => Math.round(v)).join(", ")
                        : "нет";

                    document.getElementById("overlay").src = "data:image/png;base64," + data.overlay_base64;
                    result.classList.add("show");
                } catch (err) {
                    alert("Ошибка: " + err.message);
                } finally {
                    loading.style.display = "none";
                }
            });
        </script>
    </body>
    </html>
    """


# === 5. Точка входа ===
if __name__ == "__main__":
    import uvicorn
    print(f"\nСервис будет доступен на http://localhost:{PORT}")
    print(f"Swagger: http://localhost:{PORT}/docs")
    uvicorn.run(app, host="127.0.0.1", port=PORT)