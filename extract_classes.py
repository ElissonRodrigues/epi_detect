#!/usr/bin/env python3
"""
Script para extrair todas as classes disponíveis do dataset PascalVOC.
Analisa todos os arquivos XML na pasta de anotações e lista todas as classes únicas encontradas.
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_voc_annotation(xml_path: str) -> list:
    """
    Faz o parse de um arquivo de anotação PascalVOC e retorna as classes encontradas.

    Args:
        xml_path: Caminho para o arquivo XML de anotação.

    Returns:
        Lista com os nomes das classes encontradas no arquivo.
    """
    classes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Encontra todos os objetos na anotação
        for obj in root.findall("object"):
            name = obj.find("name")
            if name is not None and name.text:
                classes.append(name.text.strip())
    except ET.ParseError as e:
        print(f"Erro ao processar {xml_path}: {e}")
    except Exception as e:
        print(f"Erro inesperado em {xml_path}: {e}")

    return classes


def extract_all_classes(annotations_dir: str) -> dict:
    """
    Extrai todas as classes de todos os arquivos de anotação no diretório.

    Args:
        annotations_dir: Caminho para o diretório de anotações.

    Returns:
        Dicionário com contagem de cada classe.
    """
    class_counter = Counter()
    processed_files = 0
    error_files = 0

    annotations_path = Path(annotations_dir)

    # Lista todos os arquivos XML no diretório
    xml_files = list(annotations_path.glob("*.xml"))
    total_files = len(xml_files)

    print(f"\n📁 Diretório de anotações: {annotations_dir}")
    print(f"📄 Total de arquivos XML encontrados: {total_files}\n")

    for xml_file in xml_files:
        classes = parse_voc_annotation(str(xml_file))
        if classes:
            class_counter.update(classes)
            processed_files += 1
        else:
            error_files += 1

    return class_counter, processed_files, error_files, total_files


def main():
    # Diretório de anotações (caminho relativo ao script)
    script_dir = Path(__file__).parent
    annotations_dir = script_dir / "dataset" / "annotations"

    # Verifica se o diretório existe
    if not annotations_dir.exists():
        print(f"❌ Diretório de anotações não encontrado: {annotations_dir}")
        print("\nVerifique se o caminho está correto.")
        return

    print("=" * 60)
    print("🔍 EXTRATOR DE CLASSES - DATASET PASCALVOC")
    print("=" * 60)

    # Extrai as classes
    class_counter, processed, errors, total = extract_all_classes(str(annotations_dir))

    # Exibe os resultados
    print("-" * 60)
    print("📊 ESTATÍSTICAS DE PROCESSAMENTO:")
    print("-" * 60)
    print(f"✅ Arquivos processados com sucesso: {processed}")
    print(f"⚠️  Arquivos com erros ou vazios: {errors}")
    print(f"📄 Total de arquivos: {total}")

    print("\n" + "-" * 60)
    print("🏷️  CLASSES ENCONTRADAS:")
    print("-" * 60)

    if class_counter:
        # Ordena por quantidade (maior para menor)
        sorted_classes = class_counter.most_common()

        total_instances = sum(class_counter.values())

        print(f"\n{'Classe':<25} {'Quantidade':>12} {'Porcentagem':>12}")
        print("-" * 50)

        for class_name, count in sorted_classes:
            percentage = (count / total_instances) * 100
            print(f"{class_name:<25} {count:>12,} {percentage:>11.2f}%")

        print("-" * 50)
        print(f"{'TOTAL':<25} {total_instances:>12,} {'100.00%':>12}")

        print("\n" + "-" * 60)
        print("📋 LISTA SIMPLES DE CLASSES (para uso em código):")
        print("-" * 60)
        print("\nclasses = [")
        for i, (class_name, _) in enumerate(sorted_classes):
            comma = "," if i < len(sorted_classes) - 1 else ""
            print(f'    "{class_name}"{comma}')
        print("]")

        print(f"\nTotal de classes únicas: {len(class_counter)}")
    else:
        print("❌ Nenhuma classe encontrada nas anotações!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
