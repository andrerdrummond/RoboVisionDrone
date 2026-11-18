
import argparse
import math
import csv

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from caminhos import (
    DATASET,
    SIMULACOES,
    localizar_modelo,
    criar_execucao
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pesos",
    default=None
)

parser.add_argument(
    "--imagem",
    default=None
)

args = parser.parse_args()

pesos = localizar_modelo(
    "finetune",
    execucao=args.pesos
)

pasta_teste = DATASET / "images" / "test"

if args.imagem:
    from pathlib import Path
    caminho = Path(args.imagem).expanduser()

    if not caminho.is_absolute():
        from caminhos import BASE_DIR
        caminho = BASE_DIR / caminho

    caminho = caminho.resolve()
else:
    imagens = sorted(
        p for p in pasta_teste.iterdir()
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png"
        )
    )

    if not imagens:
        raise RuntimeError(
            "Nenhuma imagem de teste encontrada"
        )

    caminho = imagens[0]

if not caminho.is_file():
    raise FileNotFoundError(caminho)

modelo = YOLO(str(pesos))

resultado = modelo.predict(
    source=str(caminho),
    imgsz=640,
    conf=0.30,
    device="cpu",
    verbose=False
)[0]

imagem = cv2.imread(str(caminho))

if imagem is None:
    raise RuntimeError(
        "Erro ao carregar imagem."
    )

H, W = imagem.shape[:2]

alvos = []

for box in resultado.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    alvos.append((xc, yc))

    cv2.rectangle(
        imagem,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

# Posicao inicial do robo virtual
robo = (W * 0.05, H * 0.90)
trajetoria = [robo]

pendentes = alvos.copy()

while pendentes:
    destino = min(
        pendentes,
        key=lambda p: math.dist(robo, p)
    )

    trajetoria.append(destino)
    robo = destino
    pendentes.remove(destino)

saida = criar_execucao(
    SIMULACOES,
    "coleta"
)

# Salvar a sequencia de visitas
with open(
    saida / "trajetoria.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    escritor = csv.writer(f)

    escritor.writerow([
        "ordem",
        "x_pixel",
        "y_pixel"
    ])

    for i, (x, y) in enumerate(trajetoria):
        escritor.writerow([i, x, y])

# Criar imagem da trajetoria
imagem_rgb = cv2.cvtColor(
    imagem,
    cv2.COLOR_BGR2RGB
)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(imagem_rgb)

for i, (x, y) in enumerate(trajetoria):
    ax.scatter(
        x, y,
        color="blue" if i == 0 else "red",
        s=50
    )

    ax.text(
        x + 8, y + 8,
        "R" if i == 0 else str(i),
        fontsize=10
    )

for i in range(len(trajetoria) - 1):
    x1, y1 = trajetoria[i]
    x2, y2 = trajetoria[i + 1]

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "color": "blue",
            "lw": 1.5
        }
    )

ax.set_title(
    "RoboVisionML - Simulacao de coleta"
)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")

plt.savefig(
    saida / "simulacao_coleta.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nModelo:", pesos)
print("Imagem:", caminho)
print("Residuos detectados:", len(alvos))
print("Resultados salvos em:", saida)
