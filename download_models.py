#!/usr/bin/env python3
"""
Model downloader for ProUpscaler
Downloads GFPGAN v1.4 and Real-ESRGAN x2/x4 weights
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "codeformer.pth": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "size_mb": 375,
        "description": "CodeFormer — SOTA face restoration 2024-2025 (recomendado)"
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "size_mb": 332,
        "description": "GFPGAN v1.4 — Face restoration clásico (alternativa rápida)"
    },
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size_mb": 67,
        "description": "Real-ESRGAN x4 — Upscaling general + fondo 4x"
    },
    "RealESRGAN_x2plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "size_mb": 67,
        "description": "Real-ESRGAN x2 — Upscaling general 2x (más rápido)"
    },
}


def download_file(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  ✅ Ya existe: {dest.name}")
        return True

    print(f"  ⬇  Descargando {dest.name} ...")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            unit_divisor=1024, desc=f"    {dest.name}"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"  ✅ Descargado: {dest.name}")
        return True

    except Exception as e:
        print(f"  ❌ Error descargando {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   ProUpscaler — Descarga de Modelos  ║")
    print("╚══════════════════════════════════════╝\n")
    print(f"📁 Destino: {MODELS_DIR}\n")

    ok = 0
    for filename, info in MODELS.items():
        print(f"🤖 {info['description']} (~{info['size_mb']} MB)")
        dest = MODELS_DIR / filename
        if download_file(info["url"], dest, info["description"]):
            ok += 1
        print()

    print(f"{'─'*40}")
    print(f"✅ {ok}/{len(MODELS)} modelos listos.")
    if ok == len(MODELS):
        print("🚀 ¡Todo listo! Ejecuta run.sh para iniciar ProUpscaler.\n")
    else:
        print("⚠️  Algunos modelos fallaron. Revisa tu conexión y vuelve a intentarlo.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
