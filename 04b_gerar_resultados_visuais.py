
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from caminhos import (
    DATASET,
    DETECCOES,
    localizar_modelo,
    criar_execucao
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["scratch", "frozen", "finetune"],
    default="finetune"
)

parser.add_argument(
    "--pesos",
    default=None
)

parser.add_argument(
    "--quantidade",
    type=int,
    default=30
)

args = parser.parse_args()

pesos = localizar_modelo(
    args.modo,
    execucao=args.pesos
)

pasta_teste = DATASET / "images" / "test"

if not pasta_teste.is_dir():
    raise FileNotFoundError(pasta_teste)

saida = criar_execucao(
    DETECCOES,
    args.modo
)

modelo = YOLO(str(pesos))

imagens = sorted(
    p for p in pasta_teste.iterdir()
    if p.suffix.lower() in (
        ".jpg", ".jpeg", ".png"
    )
)

if not imagens:
    raise RuntimeError(
        "Nenhuma imagem de teste encontrada."
    )

for caminho in imagens[:args.quantidade]:
    resultado = modelo.predict(
        source=str(caminho),
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    imagem_anotada = resultado.plot()

    destino = saida / f"{caminho.stem}.jpg"

    cv2.imwrite(
        str(destino),
        imagem_anotada
    )

    print(
        caminho.name,
        "Deteccoes:",
        len(resultado.boxes)
    )

print("\nModelo utilizado:", pesos)
print("Imagens geradas em:", saida)
