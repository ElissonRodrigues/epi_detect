#!/usr/bin/env python3
"""
Script para anotação automática de Pessoas usando o modelo pré-treinado (COCO).

Percorre as imagens do dataset e adiciona anotações de 'person'
detectadas pelo modelo pré-treinado do torchvision.

As anotações existentes (como EPIs) são mantidas.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms import functional as F
from tqdm import tqdm

# Configurações
CONFIDENCE_THRESHOLD = 0.5
# COCO classes: 1 is person
PERSON_CLASS_ID = 1
PERSON_CLASS_NAME = "person"


def load_model(device: torch.device) -> torch.nn.Module:
    """Carrega o modelo Faster R-CNN pré-treinado no COCO."""
    print("Carregando modelo pré-treinado (COCO)...")

    # Carregar modelo com pesos defaults (COCO)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    model = model.to(device).eval()
    print("Modelo carregado com sucesso!")
    return model


def detect_persons(
    model: torch.nn.Module, image_path: str, device: torch.device
) -> list[dict]:
    """
    Detecta pessoas em uma imagem.

    Returns:
        Lista de detecções com 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'
    """
    # Carregar imagem
    image = cv2.imread(image_path)
    if image is None:
        return []

    # Converter para tensor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

    # Detectar
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    detections = []
    for i, score in enumerate(predictions["scores"].tolist()):
        if score > CONFIDENCE_THRESHOLD:
            label_id = predictions["labels"][i].item()
            # COCO class 1 is person
            if label_id == PERSON_CLASS_ID:
                box = predictions["boxes"][i].cpu().numpy().astype(int)
                detections.append(
                    {
                        "class": PERSON_CLASS_NAME,
                        "xmin": int(box[0]),
                        "ymin": int(box[1]),
                        "xmax": int(box[2]),
                        "ymax": int(box[3]),
                        "score": score,
                    }
                )

    return detections


def get_existing_boxes(xml_path: str) -> list[dict]:
    """Retorna os bounding boxes existentes no XML."""
    boxes = []
    if not os.path.exists(xml_path):
        return boxes

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name")
            bndbox = obj.find("bndbox")
            if name is not None and bndbox is not None:
                boxes.append(
                    {
                        "class": name.text.strip() if name.text else "",
                        "xmin": int(bndbox.find("xmin").text),  # type: ignore
                        "ymin": int(bndbox.find("ymin").text),  # type: ignore
                        "xmax": int(bndbox.find("xmax").text),  # type: ignore
                        "ymax": int(bndbox.find("ymax").text),  # type: ignore
                    }
                )
    except Exception:
        pass
    return boxes


def calculate_iou(box1: dict, box2: dict) -> float:
    """Calcula IoU entre duas boxes."""
    x1 = max(box1["xmin"], box2["xmin"])
    y1 = max(box1["ymin"], box2["ymin"])
    x2 = min(box1["xmax"], box2["xmax"])
    y2 = min(box1["ymax"], box2["ymax"])

    if x2 < x1 or y2 < y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1["xmax"] - box1["xmin"]) * (box1["ymax"] - box1["ymin"])
    box2_area = (box2["xmax"] - box2["xmin"]) * (box2["ymax"] - box2["ymin"])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_duplicate(
    new_box: dict, existing_boxes: list[dict], iou_threshold: float = 0.5
) -> bool:
    """Verifica se a nova box é duplicata de alguma existente."""
    for existing in existing_boxes:
        if existing["class"] == new_box["class"]:
            if calculate_iou(new_box, existing) > iou_threshold:
                return True
    return False


def create_base_xml(
    image_path: str, width: int, height: int, depth: int = 3
) -> ET.Element:
    """Cria estrutura básica do XML."""
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = os.path.basename(os.path.dirname(image_path))

    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)

    path = ET.SubElement(root, "path")
    path.text = image_path

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    d = ET.SubElement(size, "depth")
    d.text = str(depth)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    return root


def add_annotations_to_xml(
    xml_path: str, image_path: str, detections: list[dict], dry_run: bool = False
) -> int:
    """
    Adiciona novas anotações ao arquivo XML.
    Se não existir, cria um novo.
    """
    if not detections:
        return 0

    created_new = False
    try:
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
        else:
            # Criar novo XML
            img = cv2.imread(image_path)
            if img is None:
                return 0
            h, w, c = img.shape
            root = create_base_xml(image_path, w, h, c)
            tree = ET.ElementTree(root)
            created_new = True
    except Exception as e:
        print(f"Erro ao ler/criar XML {xml_path}: {e}")
        return 0

    # Pegar boxes existentes para evitar duplicatas
    existing_boxes = get_existing_boxes(xml_path) if not created_new else []

    added_count = 0
    for det in detections:
        # Verificar se não é duplicata
        if is_duplicate(det, existing_boxes):
            continue

        # Criar elemento object
        obj = ET.SubElement(root, "object")

        name = ET.SubElement(obj, "name")
        name.text = det["class"]

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")

        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(det["xmin"])

        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(det["ymin"])

        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(det["xmax"])

        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(det["ymax"])

        # Adicionar à lista de existentes para evitar duplicatas no mesmo batch
        existing_boxes.append(det)
        added_count += 1

    if added_count > 0 and not dry_run:
        # Indentar o XML para ficar legível
        indent_xml(root)
        tree.write(xml_path, encoding="unicode")

    return added_count


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Indenta o XML para ficar legível."""
    indent = "\n" + "\t" * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore
            child.tail = indent  # type: ignore
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def process_dataset(
    images_dir: str,
    annotations_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """
    Processa todas as imagens do dataset separando diretórios de imagem e anotação.
    """
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)

    if not annotations_path.exists():
        annotations_path.mkdir(parents=True, exist_ok=True)

    # Listar todos os arquivos de imagem
    image_files = (
        list(images_path.glob("*.jpg"))
        + list(images_path.glob("*.png"))
        + list(images_path.glob("*.jpeg"))
    )

    if limit:
        image_files = image_files[:limit]

    stats = {
        "total_images": len(image_files),
        "images_with_new_annotations": 0,
        "total_annotations_added": 0,
        "errors": 0,
    }

    print(f"\n{'='*60}")
    print(f"Processando {len(image_files)} imagens...")
    print(f"Imagens: {images_dir}")
    print(f"Anotações: {annotations_dir}")
    if dry_run:
        print("⚠️  MODO DRY RUN - Nenhuma alteração será salva")
    print(f"{'='*60}\n")

    for image_path in tqdm(image_files, desc="Anotando pessoas"):
        # Definir caminho do XML
        xml_name = image_path.stem + ".xml"
        xml_path = annotations_path / xml_name

        try:
            # Detectar Pessoas
            detections = detect_persons(model, str(image_path), device)

            if detections:
                # Adicionar anotações ao XML
                added = add_annotations_to_xml(
                    str(xml_path), str(image_path), detections, dry_run
                )

                if added > 0:
                    stats["images_with_new_annotations"] += 1
                    stats["total_annotations_added"] += added

        except Exception as e:
            stats["errors"] += 1
            print(f"\nErro ao processar {image_path.name}: {e}")

    return stats


def print_stats(stats: dict) -> None:
    """Imprime estatísticas do processamento."""
    print(f"\n{'='*60}")
    print("📊 ESTATÍSTICAS DO PROCESSAMENTO")
    print(f"{'='*60}")
    print(f"📷 Total de imagens processadas: {stats['total_images']}")
    print(
        f"✅ Imagens com novas anotações de pessoa: {stats['images_with_new_annotations']}"
    )
    print(f"📝 Total de anotações adicionadas: {stats['total_annotations_added']}")
    print(f"❌ Erros: {stats['errors']}")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Anotação automática de Pessoas usando modelo pré-treinado"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="dataset/images",
        help="Diretório das imagens (default: dataset/images)",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations",
        help="Diretório das anotações (default: dataset/annotations)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold de confiança (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa sem salvar alterações (para teste)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de imagens a processar (para teste)",
    )

    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    # Verificar se diretório de imagens existe
    if not os.path.exists(args.images):
        print(f"❌ Diretório de imagens não encontrado: {args.images}")
        sys.exit(1)

    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando dispositivo: {device}")

    # Carregar modelo
    model = load_model(device)

    # Processar dataset
    stats = process_dataset(
        args.images,
        args.annotations,
        model,
        device,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    # Imprimir estatísticas
    print_stats(stats)

    if args.dry_run:
        print("💡 Para aplicar as alterações, execute novamente sem --dry-run")


if __name__ == "__main__":
    main()
