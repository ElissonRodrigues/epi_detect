import argparse
import time

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Configurações
CONFIDENCE_THRESHOLD = 0.7
# Classes do dataset EPI (devem incluir 'person' para verificar uso de EPIs)
CLASS_NAMES = ["__background__", "person", "helmet", "vest", "gloves"]


def load_models(device: torch.device, epi_model_path: str | None = None):
    """
    Carrega dois modelos:
    1. Modelo COCO pré-treinado para detectar pessoas
    2. Modelo customizado para detectar EPIs (helmet, vest, gloves)
    """
    print(f"Carregando modelos no dispositivo: {device}...")

    # COCO pré-treinado para detectar pessoas
    print("  - Carregando modelo COCO para detecção de pessoas...")
    person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    person_model = person_model.to(device).eval()

    # Customizado para EPIs
    epi_model = None
    if epi_model_path:
        print(f"  - Carregando modelo customizado para EPIs de: {epi_model_path}")
        epi_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # 4 classes: background, helmet, vest, gloves
        in_features = epi_model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        epi_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

        state_dict = torch.load(epi_model_path, map_location=device, weights_only=True)
        epi_model.load_state_dict(state_dict)
        epi_model = epi_model.to(device).eval()

    print("Modelos carregados com sucesso!")
    return person_model, epi_model


def process_predictions(predictions: dict, class_names: list[str]) -> list[tuple]:
    """Processa as predições e retorna lista de (box, label, score)."""
    return [
        (
            predictions["boxes"][i].cpu().numpy().astype(int),
            (
                class_names[label_id]
                if (label_id := predictions["labels"][i].item()) < len(class_names)
                else str(label_id)  # type: ignore
            ),
            score,
        )
        for i, score in enumerate(predictions["scores"].tolist())
        if score > CONFIDENCE_THRESHOLD
    ]


def combine_predictions(person_predictions, epi_predictions):
    """
    Combina predições de dois modelos:
    - person_predictions: do modelo COCO (classe 'person' = 1)
    - epi_predictions: do modelo customizado (helmet=1, vest=2, gloves=3)

    Retorna lista unificada de (box, label, score)
    """
    # Classes do modelo de EPIs
    EPI_CLASSES = ["__background__", "helmet", "vest", "gloves"]

    combined = []

    # Adicionar detecções de pessoas
    if person_predictions:
        for i, score in enumerate(person_predictions["scores"].tolist()):
            if score > CONFIDENCE_THRESHOLD:
                label_id = person_predictions["labels"][i].item()
                if label_id == 1:  # person no COCO
                    box = person_predictions["boxes"][i].cpu().numpy().astype(int)
                    combined.append((box, "person", score))

    # Adicionar detecções de EPIs
    if epi_predictions:
        for i, score in enumerate(epi_predictions["scores"].tolist()):
            if score > CONFIDENCE_THRESHOLD:
                label_id = epi_predictions["labels"][i].item()
                if 0 < label_id < len(EPI_CLASSES):
                    box = epi_predictions["boxes"][i].cpu().numpy().astype(int)
                    label = EPI_CLASSES[label_id]
                    combined.append((box, label, score))

    return combined


def calculate_iou(box1, box2):
    """Calcula o IoU (Intersection over Union) entre duas caixas delimitadoras."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calcular área de interseção
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calcular áreas das caixas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # IoU = área de interseção / área de união
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def check_epi_usage(results: list[tuple], iou_threshold: float = 0.1):
    """
    Verifica se pessoas estão usando EPIs.
    Retorna dict com informações sobre uso de EPIs por pessoa.
    """
    persons = [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "person"]
    epis = {
        "helmet": [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "helmet"],
        "vest": [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "vest"],
        "gloves": [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "gloves"],
    }

    person_epi_status = []

    for person_box, person_idx in persons:
        status = {
            "person_idx": person_idx,
            "box": person_box,
            "helmet": False,
            "vest": False,
            "gloves": False,
        }

        # Verificar cada tipo de EPI
        for epi_type, epi_list in epis.items():
            for epi_box, epi_idx in epi_list:
                iou = calculate_iou(person_box, epi_box)
                if iou > iou_threshold:
                    status[epi_type] = True
                    break

        person_epi_status.append(status)

    return person_epi_status


def draw_detections(frame, results: list[tuple], epi_status: list[dict] | None = None) -> None:
    """Desenha as detecções no frame com status de EPI."""

    # Criar mapeamento de pessoa para status de EPI
    person_status_map = {}
    epis_in_use = set()  # Índices de EPIs que estão sendo usados

    if epi_status:
        for status in epi_status:
            person_idx = status["person_idx"]
            person_status_map[person_idx] = status

            # Marcar EPIs que estão sendo usados (baseado em IoU)
            # Vamos re-calcular quais EPIs estão próximos desta pessoa
            person_box = status["box"]
            for idx, (box, label, _) in enumerate(results):
                if label in ["helmet", "vest", "gloves"]:
                    iou = calculate_iou(person_box, box)
                    if iou > 0.1:  # Mesmo threshold usado em check_epi_usage
                        epis_in_use.add(idx)

    # Desenhar todas as detecções
    for idx, (box, label, score) in enumerate(results):
        x1, y1, x2, y2 = box

        # Determinar cor baseada no tipo de objeto e status de EPI
        if label == "person":
            # Determinar cor da pessoa baseado no status de EPIs
            if idx in person_status_map:
                status = person_status_map[idx]
                has_helmet = status["helmet"]
                has_vest = status["vest"]
                has_gloves = status["gloves"]

                # Sistema de cores graduado
                if has_helmet and has_vest and has_gloves:
                    color = (0, 100, 0)  # Verde escuro - Todos os EPIs
                elif has_helmet and has_vest:
                    color = (0, 255, 0)  # Verde claro - Capacete + Colete
                elif has_helmet:
                    color = (0, 255, 255)  # Amarelo - Apenas capacete
                else:
                    color = (0, 0, 255)  # Vermelho - Sem capacete ou nenhum EPI
            else:
                color = (0, 255, 255)  # Amarelo - Pessoa sem status
        elif label in ["helmet", "vest", "gloves"]:
            # EPIs sendo usados = laranja, EPIs soltos = cinza claro
            if idx in epis_in_use:
                color = (255, 165, 0)  # Laranja - EPI em uso
            else:
                color = (180, 180, 180)  # Cinza claro - EPI não usado
        else:
            color = (255, 0, 0)  # Azul para outros

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Desenhar status de EPI para pessoas
    if epi_status:
        frame_height, frame_width = frame.shape[:2]

        for status in epi_status:
            x1, y1, x2, y2 = status["box"]

            # Criar texto de status
            missing_epis = [epi for epi in ["helmet", "vest", "gloves"] if not status[epi]]

            # Cor baseada no compliance
            all_epis = len(missing_epis) == 0
            status_color = (0, 255, 0) if all_epis else (0, 0, 255)

            # Preparar texto
            status_text = "[OK] Usando EPIs" if all_epis else f"[X] Faltam: {', '.join(missing_epis)}"

            # Calcular posição do texto (abaixo da pessoa, centralizado)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)

            # Posição central abaixo da box da pessoa
            text_x = x1 + (x2 - x1) // 2 - text_width // 2
            text_y = y2 + 35

            # Verificar limites da imagem e ajustar se necessário
            if text_x < 5:
                text_x = 5
            if text_x + text_width > frame_width - 5:
                text_x = frame_width - text_width - 5
            if text_y + text_height > frame_height - 5:
                text_y = y1 - 15

            # Desenhar fundo preto para o texto
            padding = 5
            bg_x1 = text_x - padding
            bg_y1 = text_y - text_height - padding
            bg_x2 = text_x + text_width + padding
            bg_y2 = text_y + baseline + padding

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), status_color, 2)

            # Desenhar linha conectando a pessoa ao status
            center_person = (x1 + (x2 - x1) // 2, y2)
            center_text = (text_x + text_width // 2, bg_y1)
            cv2.line(frame, center_person, center_text, status_color, 2)

            # Desenhar texto
            cv2.putText(frame, status_text, (text_x, text_y), font, font_scale, status_color, thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecção de EPIs com Faster R-CNN")
    parser.add_argument("--source", type=str, default="0", help="Caminho do vídeo ou índice da webcam (padrão: 0)")
    parser.add_argument("--model", type=str, default=None, help="Caminho para o modelo de EPIs treinado (.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar ambos os modelos
    person_model, epi_model = load_models(device, args.model)

    source = int(args.source) if args.source.isdigit() else args.source

    # Detectar se é imagem estática
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif")
    is_image = isinstance(source, str) and source.lower().endswith(image_extensions)

    if is_image:
        # Processar imagem estática
        frame = cv2.imread(source)
        if frame is None:
            print(f"Erro: Não foi possível carregar a imagem '{source}'")
            return

        print(f"Processando imagem: {source}")

        # Inferência com ambos os modelos
        img_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.perf_counter()

            # Predições de pessoas (modelo COCO)
            person_predictions = person_model(img_tensor)[0]

            # Predições de EPIs (modelo customizado)
            epi_predictions = epi_model(img_tensor)[0] if epi_model is not None else None

            inference_time = time.perf_counter() - start_time

        # Combinar detecções dos dois modelos
        results = combine_predictions(person_predictions, epi_predictions)

        # Verificar uso de EPIs
        epi_status = check_epi_usage(results)

        # Desenhar detecções com status de EPIs
        draw_detections(frame, results, epi_status)

        # Adicionar tempo de inferência
        cv2.putText(
            frame,
            f"Tempo: {inference_time*1000:.1f}ms",
            (frame.shape[1] - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        print(f"Detecções encontradas: {len(results)}")
        for box, label, score in results:
            print(f"  - {label}: {score:.2f} [x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}]")

        # Criar janela com configurações adequadas
        window_name = "Faster R-CNN - Detecção"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # Debug: verificar se a imagem tem dados válidos
        print(f"Shape do frame: {frame.shape}, Min: {frame.min()}, Max: {frame.max()}")

        # Calcular tamanho da janela respeitando proporção da imagem
        height, width = frame.shape[:2]
        max_width = 1200
        max_height = 900

        # Calcular escala para caber na tela
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # Não aumentar se já couber

        window_width = int(width * scale)
        window_height = int(height * scale)

        cv2.imshow(window_name, frame)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.waitKey(1)  # Pequeno delay para renderizar a janela
        print("Pressione qualquer tecla para sair...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Processar vídeo ou webcam
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a fonte '{source}'")
            return

        prev_time = time.perf_counter()
        print("Iniciando detecção. Pressione 'q' para sair.")

        while cap.isOpened():
            if not (ret := cap.read())[0]:
                break

            frame = ret[1]  # type: ignore

            # Inferência com ambos os modelos
            img_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                # Predições de pessoas (modelo COCO)
                person_predictions = person_model(img_tensor)[0]

                # Predições de EPIs (modelo customizado)
                epi_predictions = epi_model(img_tensor)[0] if epi_model is not None else None

            # Combinar detecções dos dois modelos
            results = combine_predictions(person_predictions, epi_predictions)

            # Verificar uso de EPIs
            epi_status = check_epi_usage(results)

            # Desenhar detecções com status de EPIs
            draw_detections(frame, results, epi_status)

            # Cálculo de FPS
            curr_time = time.perf_counter()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.imshow("Faster R-CNN", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
