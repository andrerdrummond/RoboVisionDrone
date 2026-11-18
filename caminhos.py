
from pathlib import Path
from datetime import datetime

# Pasta principal do projeto
BASE_DIR = Path(__file__).resolve().parent

# Dataset
DADOS = BASE_DIR / "dados"
DATASET = DADOS / "yolo"
DATASET_YAML = BASE_DIR / "dataset.yaml"

# Resultados
RESULTADOS = BASE_DIR / "resultados"
AVALIACOES = RESULTADOS / "avaliacoes"
DETECCOES = RESULTADOS / "deteccoes"
SIMULACOES = RESULTADOS / "simulacoes"

# Modelos atuais e antigos
RAIZES_MODELOS = [
    RESULTADOS,
    BASE_DIR / "runs" / "detect" / "resultados",
    BASE_DIR / "runs" / "detect",
]

def criar_execucao(pasta_base, nome):
    pasta_base = Path(pasta_base).resolve()
    pasta_base.mkdir(parents=True, exist_ok=True)

    horario = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = pasta_base / f"{nome}_{horario}"

    contador = 1
    while destino.exists():
        destino = pasta_base / f"{nome}_{horario}_{contador:02d}"
        contador += 1

    destino.mkdir(parents=True)
    return destino


def localizar_modelo(modo, execucao=None):
    if execucao:
        caminho = Path(execucao).expanduser()

        if not caminho.is_absolute():
            caminho = BASE_DIR / caminho

        caminho = caminho.resolve()

        if caminho.is_file() and caminho.suffix == ".pt":
            return caminho

        peso = caminho / "weights" / "best.pt"
        if peso.is_file():
            return peso

        raise FileNotFoundError(caminho)

    encontrados = []

    for raiz in RAIZES_MODELOS:
        if not raiz.is_dir():
            continue

        pastas = [raiz / modo]
        pastas.extend(raiz.glob(f"{modo}_*"))

        for pasta in pastas:
            peso = pasta / "weights" / "best.pt"
            if peso.is_file():
                encontrados.append(peso.resolve())

    encontrados = list(dict.fromkeys(encontrados))

    if len(encontrados) == 1:
        return encontrados[0]

    if len(encontrados) > 1:
        raise RuntimeError(
            "Multiplos modelos encontrados. "
            "Informe --pesos:\n"
            + "\n".join(map(str, encontrados))
        )

    raise FileNotFoundError(
        f"Nenhum modelo encontrado para {modo}"
    )
