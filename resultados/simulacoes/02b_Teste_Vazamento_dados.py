import hashlib
from pathlib import Path
from collections import defaultdict

base = Path(r"C:\Users\andrerdrummond\Downloads\RoboVisionDrone\dados\yolo")
splits = ["train", "val", "test"]

hashes = defaultdict(list)
prefixos = defaultdict(set)

for split in splits:
    pasta = base / "images" / split
    if not pasta.exists():
        print(f"Pasta não encontrada: {pasta}")
        continue
    
    arquivos = list(pasta.glob("*.*"))
    print(f"Total em {split}: {len(arquivos)} imagens")
    
    for img in arquivos:
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            # 1. Verifica arquivo exatamente igual (MD5)
            h = hashlib.md5(img.read_bytes()).hexdigest()
            hashes[h].append((split, img.name))
            
            # 2. Verifica prefixo antes de sufixos do Roboflow (.rf.) ou numeração de frame
            nome_base = img.name.split(".rf.")[0]
            prefixos[nome_base].add(split)

exatas = {k: v for k, v in hashes.items() if len({s for s, _ in v}) > 1}
mesmo_nome = {k: v for k, v in prefixos.items() if len(v) > 1}

print(f"\nImagens 100% idênticas entre splits diferentes: {len(exatas)}")
print(f"Imagens com mesmo nome-base (ex: cópias com augmentation) entre splits: {len(mesmo_nome)}")

if mesmo_nome:
    exemplo = list(mesmo_nome.items())[0]
    print(f"Exemplo de vazamento encontrado: {exemplo[0]} aparece em {exemplo[1]}")
