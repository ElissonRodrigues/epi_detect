"""
Flask application para monitoramento de EPIs em tempo real via WebSocket.
"""

# Monkey patching DEVE vir primeiro
import eventlet
eventlet.monkey_patch()

import base64
import os
import sys
import time

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Adicionar diretório pai ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.detector import EPIDetector

app = Flask(__name__)
app.config["SECRET_KEY"] = "epi-detect-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Detector global (inicializado no startup)
detector: EPIDetector | None = None


def get_model_path() -> str | None:
    """Retorna o caminho do modelo EPI se existir."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelo_epi.pth"
    )
    return model_path if os.path.exists(model_path) else None


@app.route("/")
def index():
    """Página principal com interface de monitoramento."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    """Quando um cliente conecta."""
    print("Cliente conectado")
    emit("status", {"message": "Conectado ao servidor de detecção de EPIs"})


@socketio.on("disconnect")
def handle_disconnect():
    """Quando um cliente desconecta."""
    print("Cliente desconectado")


@socketio.on("frame")
def handle_frame(data):
    """
    Recebe um frame do cliente, processa com o detector e retorna resultado.

    Args:
        data: Dict com 'image' contendo a imagem em base64
    """
    global detector
    
    start_time = time.time()
    print(f"Frame recebido - tamanho: {len(str(data.get('image', '')))}")

    if detector is None:
        emit("error", {"message": "Detector não inicializado"})
        return

    try:
        # Decodificar imagem base64
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit("error", {"message": "Erro ao decodificar imagem"})
            return

        # Detectar EPIs
        results, epi_status = detector.detect(frame)

        # Desenhar detecções no frame
        annotated_frame = draw_detections(frame, results, epi_status)

        # Codificar frame anotado para base64
        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_base64 = base64.b64encode(buffer).decode("utf-8")

        # Preparar resposta com detecções e status
        response = {
            "image": f"data:image/jpeg;base64,{annotated_base64}",
            "detections": [
                {"box": box.tolist(), "label": label, "score": score}
                for box, label, score in results
            ],
            "epi_status": epi_status,
            "total_persons": len(epi_status),
            "compliant": sum(
                1
                for s in epi_status
                if s["helmet"] and s["vest"] and s["gloves"]
            ),
            "timestamp": data.get("timestamp"),
        }

        elapsed = time.time() - start_time
        print(f"Frame processado em {elapsed*1000:.0f}ms - {len(results)} detecções")
        emit("result", response)

    except Exception as e:
        import traceback
        print(f"Erro ao processar frame: {e}")
        traceback.print_exc()
        emit("error", {"message": str(e)})


def draw_detections(
    frame: np.ndarray, results: list[tuple], epi_status: list[dict]
) -> np.ndarray:
    """Desenha detecções e status no frame."""
    annotated = frame.copy()

    # Mapeamento de pessoa para status
    person_status_map = {s["person_idx"]: s for s in epi_status}

    # Cores para cada tipo
    colors = {
        "person_ok": (0, 200, 0),      # Verde - todos EPIs
        "person_partial": (0, 200, 255),  # Laranja - alguns EPIs
        "person_none": (0, 0, 255),    # Vermelho - sem EPIs
        "helmet": (255, 200, 0),       # Azul claro
        "vest": (255, 100, 200),       # Rosa
        "gloves": (100, 255, 200),     # Ciano
    }

    for idx, (box, label, score) in enumerate(results):
        x1, y1, x2, y2 = box

        if label == "person":
            # Determinar cor baseada no status de EPIs
            if idx in person_status_map:
                status = person_status_map[idx]
                epi_count = sum([status["helmet"], status["vest"], status["gloves"]])
                if epi_count == 3:
                    color = colors["person_ok"]
                elif epi_count > 0:
                    color = colors["person_partial"]
                else:
                    color = colors["person_none"]
            else:
                color = colors["person_partial"]

            # Desenhar bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Status text
            if idx in person_status_map:
                status = person_status_map[idx]
                missing = []
                if not status["helmet"]:
                    missing.append("🪖")
                if not status["vest"]:
                    missing.append("🦺")
                if not status["gloves"]:
                    missing.append("🧤")

                if missing:
                    status_text = f"Falta: {' '.join(missing)}"
                else:
                    status_text = "OK - Todos EPIs"

                # Background para texto
                (tw, th), _ = cv2.getTextSize(
                    status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated, (x1, y2 + 5), (x1 + tw + 10, y2 + th + 15), color, -1
                )
                cv2.putText(
                    annotated,
                    status_text,
                    (x1 + 5, y2 + th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
        else:
            # EPI items
            color = colors.get(label, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label}: {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return annotated


def init_detector():
    """Inicializa o detector de EPIs."""
    global detector
    model_path = get_model_path()
    print(f"Inicializando detector com modelo: {model_path}")
    detector = EPIDetector(model_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Sistema de Monitoramento de EPIs em Tempo Real")
    print("=" * 60)

    init_detector()

    print("\nServidor iniciando em http://localhost:5000")
    print("Pressione Ctrl+C para encerrar\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
