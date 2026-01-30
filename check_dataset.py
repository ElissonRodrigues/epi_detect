import os
from pathlib import Path


def get_stems(directory, extensions):
    stems = set()
    path = Path(directory)
    for ext in extensions:
        for file in path.glob(f"*{ext}"):
            stems.add(file.stem)
    return stems


def main():
    images_dir = "dataset/images"
    annotations_dir = "dataset/annotations"

    print(f"Verificando integridade do dataset...")
    print(f"Imagens: {images_dir}")
    print(f"Anotações: {annotations_dir}")

    image_stems = get_stems(images_dir, [".jpg", ".jpeg", ".png"])
    xml_stems = get_stems(annotations_dir, [".xml"])

    print(f"\nTotal de Imagens encontradas: {len(image_stems)}")
    print(f"Total de Anotações encontradas: {len(xml_stems)}")

    # Imagens sem anotação
    images_without_xml = image_stems - xml_stems
    print(f"\n[!] Imagens SEM anotação ({len(images_without_xml)}):")
    for i, stem in enumerate(sorted(list(images_without_xml))):
        if i < 10:
            print(f"  - {stem}")
        elif i == 10:
            print(f"  ... e mais {len(images_without_xml) - 10}")

    # Anotações sem imagem
    xml_without_images = xml_stems - image_stems
    print(f"\n[!] Anotações SEM imagem ({len(xml_without_images)}):")

    orphaned_files = []

    for i, stem in enumerate(sorted(list(xml_without_images))):
        if i < 10:
            print(f"  - {stem}")
        elif i == 10:
            print(f"  ... e mais {len(xml_without_images) - 10}")
        # Add to list for deletion
        # Note: We need to reconstruct the full path. Since stem doesn't have extension, we add .xml
        orphaned_files.append(os.path.join(annotations_dir, f"{stem}.xml"))

    if orphaned_files:
        print(f"\nRemovendo {len(orphaned_files)} arquivos XML órfãos...")
        for file_path in orphaned_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                print(f"Erro ao remover {file_path}: {e}")
        print("✅ Limpeza concluída!")
    else:
        print("✨ Nenhum arquivo órfão para remover.")


if __name__ == "__main__":
    main()
