
from pathlib import Path
import random
import cv2

random.seed(42)

PASTA = Path("dados/yolo")
SAIDA = Path("resultados/anotacoes")
SAIDA.mkdir(parents=True, exist_ok=True)

imagens = [
    p for p in (PASTA / "images/train").iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
]

selecionadas = random.sample(
    imagens, min(10, len(imagens))
)

for caminho in selecionadas:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        continue

    H, W = imagem.shape[:2]

    rotulo = (
        PASTA / "labels/train" /
        f"{caminho.stem}.txt"
    )

    for linha in rotulo.read_text().splitlines():
        classe, xc, yc, w, h = map(
            float, linha.split()
        )

        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)

        cv2.rectangle(
            imagem,
            (x1, y1), (x2, y2),
            (0, 255, 0), 3
        )

        cv2.putText(
            imagem,
            "Residuo",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Reduzir apenas para visualizacao
    altura, largura = imagem.shape[:2]
    escala = min(1.0, 1000 / max(altura, largura))

    imagem = cv2.resize(
        imagem,
        None,
        fx=escala,
        fy=escala
    )

    cv2.imwrite(
        str(SAIDA / caminho.name),
        imagem
    )

print("Imagens anotadas em:", SAIDA)
