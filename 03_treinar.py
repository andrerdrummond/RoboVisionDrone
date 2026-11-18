
import argparse
from ultralytics import YOLO
from pathlib import Path

from caminhos import (
    DATASET_YAML,
    RESULTADOS,
    criar_execucao
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["scratch", "frozen", "finetune"],
    required=True
)

parser.add_argument(
    "--epocas",
    type=int,
    default=30
)

parser.add_argument(
    "--resolucao",
    type=int,
    default=640
)

parser.add_argument(
    "--batch",
    type=int,
    default=4
)

parser.add_argument(
    "--device",
    default="cpu"
)

args = parser.parse_args()

if not DATASET_YAML.is_file():
    raise FileNotFoundError(
        f"Dataset YAML nao encontrado: {DATASET_YAML}"
    )

# Inicializacao conforme o experimento
if args.modo == "scratch":
    modelo = YOLO("yolo26n.yaml")
else:
    modelo = YOLO("yolo26n.pt")

freeze = 10 if args.modo == "frozen" else 0

nome = (
    f"{args.modo}"
    f"_{args.resolucao}px"
    f"_{args.epocas}ep"
)

pasta = criar_execucao(RESULTADOS, nome)

print("Iniciando experimento:", args.modo)
print("Pasta de destino:", pasta)

print("\n" + "=" * 60)
print("DIAGNOSTICO DOS CAMINHOS")
print("=" * 60)

print("Script executado:", Path(__file__).resolve())
print("Diretorio de trabalho:", Path.cwd())
print("RESULTADOS:", RESULTADOS)
print("RESULTADOS absoluto:", RESULTADOS.resolve())
print("Pasta criada:", pasta)
print("Nome da execucao:", pasta.name)
print("Dataset:", DATASET_YAML)

print("=" * 60 + "\n")

modelo.train(
    data=str(DATASET_YAML),
    epochs=args.epocas,
    imgsz=args.resolucao,
    batch=args.batch,
    device=args.device,
    workers=0,
    seed=42,
    deterministic=True,
    freeze=freeze,
    patience=15,
    project=str(pasta.parent.resolve()),
    name=pasta.name,
    exist_ok=True,
    plots=True
)

print("\nTreinamento concluido!")
print("Resultados:", modelo.trainer.save_dir)
print(
    "Melhores pesos:",
    modelo.trainer.save_dir / "weights" / "best.pt"
)
