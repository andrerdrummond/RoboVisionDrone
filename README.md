# RoboVisionDrone

Pipeline de visão computacional para **detecção aérea de resíduos sólidos** a partir de imagens de drone (**UAVVaste**), utilizando **YOLO26n**, com auditoria de vazamento de dados, avaliação comparativa de estratégias de treinamento (*scratch*, *frozen* e *finetune*) e simulação de trajetória de coleta.

---

## Visão Geral

O projeto estrutura todo o ciclo experimental em scripts numerados e reprodutíveis (`seed = 42`):

1. Conversão do dataset **UAVVaste** (COCO) para o formato YOLO em classe única (`residuo`).
2. Inspeção visual das anotações de treino.
3. Treinamento comparativo com três estratégias de transferência de aprendizado.
4. Avaliação quantitativa no conjunto de teste (`mAP50`, `mAP50-95`, precisão e recall).
5. Geração de predições visuais e simulação de rota de coleta por vizinho mais próximo.
6. Auditoria automatizada de integridade e vazamento entre splits (*SHA-256* e *dHash*).

---

## Resultados de Referência

Avaliação no split de teste (**117 imagens / 586 instâncias**, resolução `640px`, 30 épocas):

| Estratégia | Precisão | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Finetune** *(pesos pré-treinados, rede livre)* | 0,820 | **0,659** | **0,721** | **0,402** |
| **Frozen** *(pesos pré-treinados, 10 camadas congeladas)* | **0,847** | 0,637 | 0,711 | 0,372 |
| **Scratch** *(treinamento do zero)* | 0,724 | 0,403 | 0,460 | 0,198 |

---

## Requisitos e Dependências

- **Python** `3.10+` (testado em `3.12`)
- **Git**
- **Espaço em disco:** ~6 GB livres (dataset original + cópia estruturada em YOLO)
- **Hardware:** executa em CPU por padrão (`--device cpu`); GPU NVIDIA com CUDA compatível é opcional (`--device 0`)

### Bibliotecas Python

| Biblioteca | Finalidade |
| :--- | :--- |
| `ultralytics` | Treinamento, validação e inferência com YOLO26n (instala `torch` e `torchvision`) |
| `opencv-python` | Leitura, desenho de caixas, redimensionamento e cálculo de *dHash* |
| `pillow` | Inspeção de dimensões de imagem na conversão COCO → YOLO |
| `pyyaml` | Geração e leitura do arquivo `dataset.yaml` |
| `pandas` | Consolidação e exportação de métricas em CSV |
| `matplotlib` | Renderização do mapa de trajetória na simulação de coleta |
| `numpy` | Operações matriciais e composição de comparações visuais |

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/andrerdrummond/RoboVisionDrone.git
cd RoboVisionDrone
```

### 2. Criar e ativar o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as bibliotecas

```bash
pip install --upgrade pip
pip install ultralytics opencv-python pillow pyyaml pandas matplotlib numpy
```

> **Nota sobre GPU:** para aceleração via CUDA, instale a versão do PyTorch compatível com o seu driver seguindo as instruções em [pytorch.org](https://pytorch.org/get-started/locally/) antes ou após instalar `ultralytics`.

---

## Obtenção do Dataset

A pasta `dados/` não é versionada no repositório (`.gitignore`). Baixe o dataset **UAVVaste** (~2,80 GB, 772 imagens, 3.718 anotações) no Zenodo:

- **Fonte:** [UAVVaste dataset | Zenodo (record 8214061)](https://zenodo.org/records/8214061)

Extraia o conteúdo de `UAVVasteDataset.zip` em `dados/original/` de modo que a estrutura fique:

```text
dados/
└── original/
    ├── annotations/
    │   └── annotations.json
    └── images/
        └── *.jpg
```

---

## Execução do Pipeline

Execute os scripts na raiz do projeto, na ordem abaixo.

### 1. Preparar o dataset (COCO → YOLO)

Converte as anotações para a classe `0: residuo`, divide os dados em `70%` treino / `15%` validação / `15%` teste (`540 / 115 / 117` imagens) e gera o `dataset.yaml` com o caminho absoluto local.

```bash
python 01_preparar_dataset.py
```

**Saídas:** `dados/yolo/` e `dataset.yaml`.

---

### 2. Visualizar anotações de treino

Gera 10 imagens aleatórias do conjunto de treino com as caixas delimitadoras desenhadas para conferência visual.

```bash
python 02_visualizar_anotacoes.py
```

**Saída:** `resultados/anotacoes/`.

---

### 3. Treinar o modelo

O parâmetro `--modo` é obrigatório (`scratch`, `frozen` ou `finetune`).

```bash
# Treinamento padrão (30 épocas, 640px, CPU)
python 03_treinar.py --modo finetune

# Outros modos experimentais
python 03_treinar.py --modo frozen
python 03_treinar.py --modo scratch

# Teste rápido de validação do fluxo (1 época)
python 03_treinar.py --modo finetune --epocas 1 --resolucao 640 --batch 4 --device cpu
```

**Parâmetros opcionais:** `--epocas` (padrão `30`), `--resolucao` (padrão `640`), `--batch` (padrão `4`), `--device` (padrão `cpu`).  
**Saída:** `resultados/<modo>_<resolucao>px_<epocas>ep_<timestamp>/weights/best.pt`.

> O repositório já inclui pesos pré-treinados (`best.pt`) em `resultados/` caso deseje avançar diretamente para avaliação, inferência ou simulação.

---

### 4. Avaliar no conjunto de teste

Calcula precisão, recall, `mAP50`, `mAP50-95` e gera gráficos de avaliação no split `test`.

```bash
python 04_avaliar.py --modo finetune
```

Se houver mais de um modelo salvo para o mesmo modo em `resultados/`, informe explicitamente `--pesos`:

```bash
python 04_avaliar.py --modo finetune --pesos resultados/finetune_640px_30ep_20261117_223240/weights/best.pt
```

**Saída:** `resultados/avaliacoes/<modo>_<resolucao>px_<timestamp>/metricas.csv` e gráficos do YOLO.

---

### 5. Gerar resultados visuais de detecção

Executa inferência em lote sobre imagens do conjunto de teste e salva as imagens anotadas.

```bash
python 04b_gerar_resultados_visuais.py --modo finetune --quantidade 30
```

Com pesos explícitos:
```bash
python 04b_gerar_resultados_visuais.py --modo finetune --pesos resultados/finetune_640px_30ep_20261117_223240/weights/best.pt
```

**Saída:** `resultados/deteccoes/<modo>_<timestamp>/`.

---

### 6. Simular trajetória de coleta

Detecta resíduos em uma imagem e calcula a rota de coleta do robô virtual a partir do canto inferior esquerdo usando heurística de vizinho mais próximo.

```bash
python 05_simular_coleta.py
```

Com pesos e imagem específicos:
```bash
python 05_simular_coleta.py --pesos resultados/finetune_640px_30ep_20261117_223240/weights/best.pt --imagem dados/yolo/images/test/img_000006.jpg
```

> Em ambientes sem interface gráfica (servidores/CI), defina `MPLBACKEND=Agg` antes do comando para salvar `simulacao_coleta.png` sem abrir janela interativa.

**Saídas:** `resultados/simulacoes/coleta_<timestamp>/simulacao_coleta.png` e `trajetoria.csv`.

---

### 7. Auditar vazamento de dados

Verifica nomes repetidos entre splits, duplicatas exatas (*SHA-256*), rótulos ausentes e pares visualmente semelhantes (*dHash* ≤ 5). Pode ser executado a qualquer momento após o passo 1.

```bash
python 07_verificar_vazamento.py
```

**Saídas:** relatórios CSV em `resultados/auditoria/` e pares lado a lado em `resultados/auditoria/comparacoes_visuais/`.

---

## Estrutura do Repositório

```text
RoboVisionDrone/
├── 01_preparar_dataset.py          # Conversão COCO → YOLO e split train/val/test
├── 02_visualizar_anotacoes.py      # Amostragem visual das anotações de treino
├── 03_treinar.py                   # Treinamento (scratch, frozen, finetune)
├── 04_avaliar.py                   # Avaliação quantitativa no conjunto de teste
├── 04b_gerar_resultados_visuais.py # Predições visuais em imagens de teste
├── 05_simular_coleta.py            # Simulação de rota de coleta
├── 07_verificar_vazamento.py       # Auditoria de duplicatas e similaridade entre splits
├── caminhos.py                     # Resolução centralizada de diretórios e pesos
├── dataset.yaml                    # Configuração do dataset YOLO
├── yolo26n.pt                      # Pesos base pré-treinados YOLO26n
├── dados/                          # Dataset local (ignorado pelo Git)
└── resultados/                     # Modelos, avaliações, detecções, simulações e auditoria
```
