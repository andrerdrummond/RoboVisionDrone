
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

SEED = 42
random.seed(SEED)

ORIGEM = Path("dados/original")
DESTINO = Path("dados/yolo")

# Localizar JSON COCO
jsons = list(ORIGEM.rglob("annotations.json"))

if not jsons:
    raise FileNotFoundError(
        "annotations.json nao encontrado em dados/original"
    )

if len(jsons) != 1:
    raise RuntimeError(
        f"Encontrados {len(jsons)} arquivos annotations.json. "
        "Selecione explicitamente o JSON desejado."
    )

arquivo_json = jsons[0]

with open(arquivo_json, encoding="utf-8") as f:
    coco = json.load(f)

print("Categorias COCO:")
for categoria in coco["categories"]:
    print(categoria["id"], categoria["name"])

# Neste projeto, todas as categorias de residuos
# serao reunidas na classe 'residuo'.
ids_validos = {
    c["id"] for c in coco["categories"]
}

# Indexar arquivos de imagem
extensoes = {".jpg", ".jpeg", ".png"}
arquivos = [
    p for p in ORIGEM.rglob("*")
    if p.suffix.lower() in extensoes
]

indice = defaultdict(list)

for arquivo in arquivos:
    indice[arquivo.name.lower()].append(arquivo)

# Associar anotacoes a imagens
anotacoes = defaultdict(list)

for ann in coco["annotations"]:
    if ann["category_id"] not in ids_validos:
        continue
    if ann.get("iscrowd", 0):
        continue
    anotacoes[ann["image_id"]].append(ann)

registros = []

for img in coco["images"]:
    nome = Path(img["file_name"]).name.lower()
    candidatos = indice.get(nome, [])

    if len(candidatos) != 1:
        print(
            "Imagem ausente ou nome duplicado:",
            img["file_name"]
        )
        continue

    caminho = candidatos[0]

    with Image.open(caminho) as im:
        W, H = im.size

    registros.append({
        "id": img["id"],
        "caminho": caminho,
        "largura": W,
        "altura": H,
        "anns": anotacoes[img["id"]]
    })

random.shuffle(registros)

n = len(registros)
n_treino = int(0.70 * n)
n_val = int(0.15 * n)

particoes = {
    "train": registros[:n_treino],
    "val": registros[n_treino:n_treino+n_val],
    "test": registros[n_treino+n_val:]
}

contagem = Counter()

for grupo, itens in particoes.items():
    pasta_img = DESTINO / "images" / grupo
    pasta_lbl = DESTINO / "labels" / grupo

    pasta_img.mkdir(parents=True, exist_ok=True)
    pasta_lbl.mkdir(parents=True, exist_ok=True)

    for item in itens:
        origem = item["caminho"]

        # ID evita colisao de nomes
        nome = f"img_{item['id']:06d}"

        extensao = origem.suffix.lower()
        destino_img = pasta_img / (nome + extensao)

        shutil.copy2(origem, destino_img)

        W = item["largura"]
        H = item["altura"]
        linhas = []

        for ann in item["anns"]:
            x, y, w, h = ann["bbox"]

            # Limitar a caixa aos limites da imagem
            x1 = max(0, min(W, x))
            y1 = max(0, min(H, y))
            x2 = max(0, min(W, x + w))
            y2 = max(0, min(H, y + h))

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            xc = (x1 + x2) / (2 * W)
            yc = (y1 + y2) / (2 * H)
            wn = bw / W
            hn = bh / H

            linhas.append(
                f"0 {xc:.6f} {yc:.6f} "
                f"{wn:.6f} {hn:.6f}"
            )

        (pasta_lbl / (nome + ".txt")).write_text(
            "\n".join(linhas),
            encoding="utf-8"
        )

        contagem[grupo] += 1

config = {
    "path": str(DESTINO.resolve().as_posix()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "residuo"}
}

with open("dataset.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        config,
        f,
        sort_keys=False,
        allow_unicode=True
    )

print("\nConversao concluida!")
print("JSON:", arquivo_json)
print("Imagens encontradas:", len(registros))
print("Distribuicao:", dict(contagem))
print("Arquivo criado: dataset.yaml")
