from pathlib import Path
from itertools import combinations
from collections import defaultdict
import hashlib
import csv

import cv2
import numpy as np

from caminhos import BASE_DIR, DATASET, RESULTADOS

# =========================
# CONFIGURACAO
# =========================
IMAGES_DIR = DATASET / "images"
LABELS_DIR = DATASET / "labels"

AUDITORIA_DIR = RESULTADOS / "auditoria"
VISUAIS_DIR = AUDITORIA_DIR / "comparacoes_visuais"

AUDITORIA_DIR.mkdir(parents=True, exist_ok=True)
VISUAIS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# dHash: quanto menor a distancia, mais parecidas as imagens
LIMIAR_DHASH = 5

# limite de pares a renderizar visualmente por categoria
MAX_VISUAIS_DUPLICATAS = 50
MAX_VISUAIS_SEMELHANTES = 50


# =========================
# FUNCOES AUXILIARES
# =========================
def hash_sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def hash_perceptual(caminho):
    """
    dHash 64 bits para detectar imagens visualmente semelhantes.
    """
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise ValueError(f"Imagem invalida: {caminho}")

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    pequena = cv2.resize(cinza, (9, 8), interpolation=cv2.INTER_AREA)
    diferencas = pequena[:, 1:] > pequena[:, :-1]

    bits = 0
    for valor in diferencas.flatten():
        bits = (bits << 1) | int(valor)

    return bits


def distancia_hash(a, b):
    return (a ^ b).bit_count()


def salvar_csv(nome, linhas, campos):
    caminho = AUDITORIA_DIR / nome
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    return caminho


def listar_imagens_split(split_dir):
    arquivos = [
        p for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSOES
    ]
    return sorted(arquivos)


def preparar_imagem_visual(caminho, titulo, largura=700, altura=500):
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return None

    imagem = cv2.resize(imagem, (largura, altura))
    cv2.putText(
        imagem,
        titulo,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return imagem


def gerar_comparacao_visual(linhas, nome_base, max_itens=50):
    total = 0

    for i, linha in enumerate(linhas[:max_itens]):
        caminho_a = Path(linha["imagem_a"])
        caminho_b = Path(linha["imagem_b"])

        img_a = preparar_imagem_visual(
            caminho_a,
            f"{linha['split_a']} | {caminho_a.name}"
        )
        img_b = preparar_imagem_visual(
            caminho_b,
            f"{linha['split_b']} | {caminho_b.name}"
        )

        if img_a is None or img_b is None:
            print(f"[AVISO] Nao foi possivel abrir um par: {caminho_a} | {caminho_b}")
            continue

        comparacao = np.hstack([img_a, img_b])

        texto_extra = []
        if "sha256" in linha:
            texto_extra.append("DUPLICATA EXATA")
        if "distancia_dhash" in linha:
            texto_extra.append(f"dHash={linha['distancia_dhash']}")

        if texto_extra:
            cv2.putText(
                comparacao,
                " | ".join(texto_extra),
                (15, comparacao.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        nome_saida = VISUAIS_DIR / f"{nome_base}_{i:04d}.jpg"
        cv2.imwrite(str(nome_saida), comparacao)
        total += 1

    return total


# =========================
# VERIFICACAO DE CAMINHOS
# =========================
print("=" * 70)
print("AUDITORIA DE VAZAMENTO - RoboVisionML Drone")
print("=" * 70)
print("BASE_DIR:", BASE_DIR)
print("DATASET:", DATASET)
print("IMAGES_DIR:", IMAGES_DIR)
print("LABELS_DIR:", LABELS_DIR)
print("AUDITORIA_DIR:", AUDITORIA_DIR)
print("=" * 70)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset nao encontrado: {DATASET}")

if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Pasta de imagens nao encontrada: {IMAGES_DIR}")

if not LABELS_DIR.exists():
    raise FileNotFoundError(f"Pasta de labels nao encontrada: {LABELS_DIR}")


# =========================
# LEITURA DAS IMAGENS
# =========================
registros = []
contagem_por_split = {}

print("\nLendo imagens...")

for split in SPLITS:
    pasta_split = IMAGES_DIR / split

    if not pasta_split.exists():
        raise FileNotFoundError(
            f"Pasta do split nao encontrada: {pasta_split}"
        )

    arquivos = listar_imagens_split(pasta_split)
    contagem_por_split[split] = len(arquivos)

    print(f"{split}: {len(arquivos)} imagens")

    for caminho in arquivos:
        try:
            sha = hash_sha256(caminho)
            dh = hash_perceptual(caminho)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {caminho}: {e}")
            continue

        registros.append({
            "split": split,
            "arquivo": str(caminho.resolve()),
            "nome": caminho.name,
            "sha256": sha,
            "dhash": dh
        })

if not registros:
    raise RuntimeError(
        "Nenhuma imagem foi encontrada. Verifique as pastas em dados/yolo/images."
    )


# =========================
# TESTE 1: NOMES REPETIDOS
# =========================
nomes = defaultdict(list)
for registro in registros:
    nomes[registro["nome"].lower()].append(registro)

nomes_repetidos = []

for nome, itens in nomes.items():
    for a, b in combinations(itens, 2):
        if a["split"] == b["split"]:
            continue
        nomes_repetidos.append({
            "split_a": a["split"],
            "split_b": b["split"],
            "imagem_a": a["arquivo"],
            "imagem_b": b["arquivo"]
        })


# =========================
# TESTE 2: DUPLICATAS EXATAS
# =========================
hashes = defaultdict(list)
for registro in registros:
    hashes[registro["sha256"]].append(registro)

duplicatas = []

for sha, itens in hashes.items():
    for a, b in combinations(itens, 2):
        if a["split"] == b["split"]:
            continue
        duplicatas.append({
            "split_a": a["split"],
            "split_b": b["split"],
            "imagem_a": a["arquivo"],
            "imagem_b": b["arquivo"],
            "sha256": sha
        })


# =========================
# TESTE 3: IMAGENS SEMELHANTES
# =========================
similares = []

for i, a in enumerate(registros):
    for b in registros[i + 1:]:
        if a["split"] == b["split"]:
            continue

        # ignorar duplicatas exatas aqui
        if a["sha256"] == b["sha256"]:
            continue

        distancia = distancia_hash(a["dhash"], b["dhash"])

        if distancia <= LIMIAR_DHASH:
            similares.append({
                "split_a": a["split"],
                "split_b": b["split"],
                "imagem_a": a["arquivo"],
                "imagem_b": b["arquivo"],
                "distancia_dhash": distancia
            })

similares.sort(key=lambda x: x["distancia_dhash"])


# =========================
# TESTE 4: IMAGENS SEM LABEL
# =========================
sem_labels = []

for registro in registros:
    imagem = Path(registro["arquivo"])
    label = LABELS_DIR / registro["split"] / f"{imagem.stem}.txt"

    if not label.is_file():
        sem_labels.append({
            "split": registro["split"],
            "imagem": str(imagem),
            "label_esperado": str(label)
        })


# =========================
# SALVAR CSVs
# =========================
campos_pares = ["split_a", "split_b", "imagem_a", "imagem_b"]

csv_nomes = salvar_csv(
    "nomes_repetidos.csv",
    nomes_repetidos,
    campos_pares
)

csv_dup = salvar_csv(
    "duplicatas_exatas.csv",
    duplicatas,
    campos_pares + ["sha256"]
)

csv_sim = salvar_csv(
    "imagens_semelhantes.csv",
    similares,
    campos_pares + ["distancia_dhash"]
)

csv_labels = salvar_csv(
    "imagens_sem_labels.csv",
    sem_labels,
    ["split", "imagem", "label_esperado"]
)


# =========================
# GERAR IMAGENS VISUAIS
# =========================
print("\nGerando comparacoes visuais...")

qtd_vis_dup = gerar_comparacao_visual(
    duplicatas,
    "duplicatas_exatas",
    MAX_VISUAIS_DUPLICATAS
)

qtd_vis_sim = gerar_comparacao_visual(
    similares,
    "imagens_semelhantes",
    MAX_VISUAIS_SEMELHANTES
)


# =========================
# RELATORIO FINAL
# =========================
print("\n" + "=" * 70)
print("RELATORIO FINAL DA AUDITORIA")
print("=" * 70)
print("Contagem de imagens por split:")
for split in SPLITS:
    print(f"  {split}: {contagem_por_split.get(split, 0)}")

print("\nResultados:")
print("  Nomes repetidos entre splits :", len(nomes_repetidos))
print("  Duplicatas exatas            :", len(duplicatas))
print("  Imagens semelhantes          :", len(similares))
print("  Imagens sem labels           :", len(sem_labels))

print("\nArquivos gerados:")
print(" ", csv_nomes)
print(" ", csv_dup)
print(" ", csv_sim)
print(" ", csv_labels)

print("\nComparacoes visuais geradas:")
print("  Duplicatas exatas :", qtd_vis_dup)
print("  Imagens semelhantes :", qtd_vis_sim)
print("  Pasta visual :", VISUAIS_DIR)

if duplicatas:
    print("\n[ALERTA] Foram encontradas duplicatas exatas entre splits.")
else:
    print("\n[OK] Nenhuma duplicata exata entre splits.")

if similares:
    print("[ATENCAO] Foram encontradas imagens visualmente semelhantes.")
else:
    print("[OK] Nenhuma imagem semelhante dentro do limiar configurado.")

if sem_labels:
    print("[ATENCAO] Existem imagens sem arquivo de label correspondente.")
else:
    print("[OK] Todas as imagens possuem arquivo de label correspondente.")

print("=" * 70)