
import argparse
import pandas as pd
from ultralytics import YOLO

from caminhos import (
    DATASET_YAML,
    AVALIACOES,
    localizar_modelo,
    criar_execucao
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["scratch", "frozen", "finetune"],
    required=True
)

parser.add_argument(
    "--pesos",
    default=None,
    help="Caminho para best.pt ou pasta da execucao"
)

parser.add_argument(
    "--resolucao",
    type=int,
    default=640
)

args = parser.parse_args()

if not DATASET_YAML.is_file():
    raise FileNotFoundError(DATASET_YAML)

pesos = localizar_modelo(
    args.modo,
    execucao=args.pesos
)

pasta_saida = criar_execucao(
    AVALIACOES,
    f"{args.modo}_{args.resolucao}px"
)

print("Modelo:", pesos)
print("Resultados:", pasta_saida)

modelo = YOLO(str(pesos))

metricas = modelo.val(
    data=str(DATASET_YAML),
    split="test",
    imgsz=args.resolucao,
    batch=4,
    device="cpu",
    workers=0,
    plots=True,
    project=str(AVALIACOES),
    name=pasta_saida.name,
    exist_ok=True
)

dados = {
    "modelo": args.modo,
    "pesos": str(pesos),
    "resolucao": args.resolucao,
    "precisao": metricas.box.mp,
    "recall": metricas.box.mr,
    "map50": metricas.box.map50,
    "map50_95": metricas.box.map
}

df = pd.DataFrame([dados])

arquivo_csv = (
    pasta_saida / "metricas.csv"
)

df.to_csv(arquivo_csv, index=False)

print("\nRESULTADOS:")
print(df.to_string(index=False))
print("\nCSV salvo em:", arquivo_csv)
