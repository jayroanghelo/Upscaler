#!/usr/bin/env python3
"""
ProUpscaler v3.0
Stack: FastAPI + Spandrel + PyTorch + Gemini AI opcional
Python 3.10+.
"""

import os, uuid, shutil, zipfile, threading, traceback, json, time
from pathlib import Path
from typing import Dict, List, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

# ─── Directorios ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
THUMBS_DIR = BASE_DIR / "thumbs"
for d in [STATIC_DIR, MODELS_DIR, UPLOAD_DIR, OUTPUT_DIR, THUMBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="ProUpscaler", version="3.0.0")

jobs:           Dict[str, Any] = {}
uploaded_files: Dict[str, Any] = {}
jobs_lock = threading.Lock()
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

class JobCancelledException(Exception): pass

def check_cancel(job_id: str):
    if job_id:
        with jobs_lock:
            if jobs.get(job_id, {}).get("cancelled"):
                raise JobCancelledException("Proceso cancelado por el usuario")

# Gemini key (persiste en memoria de sesión)
_gemini_key: str = ""

# ─── Housekeeper (Limpieza Automática) ───────────────────────────────────────
def cleanup_old_files(max_age_hours: int = 12):
    """Elimina archivos de uploads, output y thumbs que tengan más de X horas."""
    now = time.time()
    max_age_sec = max_age_hours * 3600
    cleaned_count = 0
    
    for folder in [UPLOAD_DIR, OUTPUT_DIR, THUMBS_DIR]:
        if not folder.exists(): continue
        for item in folder.iterdir():
            try:
                # Si es un directorio (como en OUTPUT_DIR/job_id) o un archivo
                mtime = item.stat().st_mtime
                if (now - mtime) > max_age_sec:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    cleaned_count += 1
            except Exception:
                pass
    if cleaned_count:
        print(f"  🧹 Housekeeper: {cleaned_count} archivos/carpetas antiguos eliminados.")

def start_housekeeper():
    """Lanza el hilo de limpieza periódica."""
    def run_forever():
        while True:
            cleanup_old_files(max_age_hours=12)
            time.sleep(3600) # Ejecutar cada hora
            
    t = threading.Thread(target=run_forever, daemon=True)
    t.start()

# ─── Logging por job ─────────────────────────────────────────────────────────
def jlog(job_id: str, level: str, msg: str):
    """Agrega un mensaje de log al job y lo imprime en consola."""
    ts = time.strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "msg": msg}
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].setdefault("logs", []).append(entry)
    icon = {"info": "ℹ", "ok": "✅", "warn": "⚠️", "error": "❌"}.get(level, "·")
    print(f"[{ts}] {icon} [{job_id[:8]}] {msg}")


# ─── Dispositivo ─────────────────────────────────────────────────────────────
def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

DEVICE = get_device()
# MPS tiene soporte limitado — usar CPU para inferencia
INFER_DEVICE = "cpu" if DEVICE == "mps" else DEVICE

# Límite de resolución de salida (evitar OOM / procesos eternos en CPU)
MAX_OUT_DIM = 4096  # px en el lado mayor


# ─── Modelos disponibles ──────────────────────────────────────────────────────
def check_models() -> Dict[str, bool]:
    return {
        "gfpgan":        (MODELS_DIR / "GFPGANv1.4.pth").exists(),
        "realesrgan_x4": (MODELS_DIR / "RealESRGAN_x4plus.pth").exists(),
        "pro_dat":       (MODELS_DIR / "4xFaceUpDAT.pth").exists(),
        "pro_skin":      (MODELS_DIR / "1x-ITF-SkinDiffDetail-Lite-v1.pth").exists(),
    }


# ─── Thumbnail ────────────────────────────────────────────────────────────────
def make_thumb(src: Path, dst: Path, size: int = 300):
    try:
        from PIL import Image
        with Image.open(src) as img:
            img.thumbnail((size, size), Image.LANCZOS)
            img.convert("RGB").save(dst, "JPEG", quality=85)
    except Exception as e:
        print(f"[thumb] {e}")


# ─── Conversiones ─────────────────────────────────────────────────────────────
def bgr_to_tensor(bgr, device="cpu"):
    import torch, numpy as np
    rgb = bgr[:, :, ::-1].copy().astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

def tensor_to_bgr(t):
    import numpy as np
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return (arr[:, :, ::-1] * 255).astype(np.uint8).copy()

def _imwrite(img, path: Path):
    import cv2
    ext = path.suffix.lower()
    params = {
        ".jpg": [cv2.IMWRITE_JPEG_QUALITY, 97],
        ".jpeg": [cv2.IMWRITE_JPEG_QUALITY, 97],
        ".webp": [cv2.IMWRITE_WEBP_QUALITY, 95],
        ".png": [cv2.IMWRITE_PNG_COMPRESSION, 1],
    }.get(ext, [])
    cv2.imwrite(str(path), img, params if params else None)


# ─── Caché de modelos Spandrel ────────────────────────────────────────────────
_spandrel_cache: Dict[str, Any] = {}

def load_spandrel(model_path: Path):
    key = str(model_path)
    if key not in _spandrel_cache:
        from spandrel import ModelLoader
        import torch
        m = ModelLoader(device=torch.device(INFER_DEVICE)).load_from_file(str(model_path))
        m.eval()
        _spandrel_cache[key] = m
    return _spandrel_cache[key]


# ─── Preparar entrada: cap de resolución para evitar procesos eternos ─────────
def _prepare_input(job_id: str, img, outscale: int) -> tuple:
    """
    Recorta la imagen de entrada si el output superaría MAX_OUT_DIM px.
    Devuelve (img_procesada, outscale_efectivo).
    """
    import cv2
    h_in, w_in = img.shape[:2]
    max_dim_in  = max(h_in, w_in)
    max_dim_out = max_dim_in * outscale

    if max_dim_out <= MAX_OUT_DIM:
        return img, outscale  # OK, nada que hacer

    # Calcular cuánto podemos pedir: new_max_dim * outscale <= MAX_OUT_DIM
    safe_input_max = MAX_OUT_DIM / outscale

    if safe_input_max < 128:
        # La imagen ya es enorme incluso a escala 1 — no upscalear, solo restaurar
        jlog(job_id, "warn",
             f"Imagen {w_in}×{h_in}px ya supera 4K incluso sin upscale. "
             f"Procesando a escala 1x (solo restauración).")
        outscale = 1
        safe_input_max = MAX_OUT_DIM

    if max_dim_in > safe_input_max:
        scale_f = safe_input_max / max_dim_in
        new_w   = max(64, int(w_in * scale_f))
        new_h   = max(64, int(h_in * scale_f))
        jlog(job_id, "warn",
             f"Output estimado {int(max_dim_out)}px > {MAX_OUT_DIM}px (4K). "
             f"Redimensionando entrada: {w_in}×{h_in} → {new_w}×{new_h}px "
             f"(salida ≈ {new_w*outscale}×{new_h*outscale}px).")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return img, outscale


# ─── Tile processing (con log de progreso por tile) ───────────────────────────
def tile_process(img_t, model, tile: int = 512, pad: int = 16,
                 job_id: str = None):
    import torch
    scale = model.scale
    b, c, h, w = img_t.shape
    oh, ow = h * scale, w * scale
    out = torch.zeros(b, c, oh, ow)
    dev = next(model.model.parameters()).device

    tiles_y = list(range(0, h, tile))
    tiles_x = list(range(0, w, tile))
    total   = len(tiles_y) * len(tiles_x)
    done    = 0

    if job_id:
        jlog(job_id, "info",
             f"Mapa de tiles: {len(tiles_x)} cols × {len(tiles_y)} filas = {total} tiles "
             f"({tile}px c/u, escala {scale}x)")

    for y in tiles_y:
        for x in tiles_x:
            y0, y1 = max(y - pad, 0), min(y + tile + pad, h)
            x0, x1 = max(x - pad, 0), min(x + tile + pad, w)
            chunk = img_t[:, :, y0:y1, x0:x1].to(dev)
            check_cancel(job_id)
            with torch.no_grad():
                chunk_out = model(chunk)
            oy0, oy1 = y * scale, min((y + tile) * scale, oh)
            ox0, ox1 = x * scale, min((x + tile) * scale, ow)
            iy0, ix0 = (y - y0) * scale, (x - x0) * scale
            iy1, ix1 = iy0 + (oy1 - oy0), ix0 + (ox1 - ox0)
            out[:, :, oy0:oy1, ox0:ox1] = chunk_out[:, :, iy0:iy1, ix0:ix1].cpu()
            done += 1
            # Log cada 5 tiles o al terminar
            if job_id and (done % 5 == 0 or done == total or done == 1):
                pct = int(done / total * 100)
                jlog(job_id, "info", f"Tiles: {done}/{total} ({pct}%)")
    return out


# ─── ESRGAN (siempre funciona) ────────────────────────────────────────────────
def run_esrgan(job_id: str, input_path: Path, output_path: Path,
               model_file: str, outscale: int, tile: int):
    import cv2
    jlog(job_id, "info", f"Cargando modelo ESRGAN: {model_file}")
    model = load_spandrel(MODELS_DIR / model_file)
    jlog(job_id, "info", f"Leyendo imagen: {input_path.name} ({input_path.stat().st_size // 1024} KB)")

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV no pudo leer la imagen: {input_path.name}")

    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        alpha = img[:, :, 3]
        img = img[:, :, :3]

    # ── Cap resolución de salida ─────────────────────────────────────────────
    img, outscale = _prepare_input(job_id, img, outscale)

    jlog(job_id, "info",
         f"Procesando tiles (tile={tile}px, scale={model.scale}x → outscale={outscale}x, "
         f"entrada {img.shape[1]}×{img.shape[0]}px)…")
    img_t = bgr_to_tensor(img)
    out_t = tile_process(img_t, model, tile, job_id=job_id)
    out   = tensor_to_bgr(out_t)

    # Ajustar al outscale deseado si difiere del scale del modelo
    h_in, w_in = img.shape[:2]
    if outscale != model.scale:
        th = int(h_in * outscale)
        tw = int(w_in * outscale)
        import cv2 as _cv
        out = _cv.resize(out, (tw, th), interpolation=_cv.INTER_LANCZOS4)

    if has_alpha:
        import cv2 as _cv, numpy as np
        alpha_up = _cv.resize(alpha, (out.shape[1], out.shape[0]),
                              interpolation=_cv.INTER_LANCZOS4)
        out = np.dstack([out, alpha_up])

    jlog(job_id, "info", f"Guardando salida → {output_path.name}")
    _imwrite(out, output_path)
    h_out, w_out = out.shape[:2]
    jlog(job_id, "ok", f"ESRGAN completado: {w_in}×{h_in} → {w_out}×{h_out}")


# ─── Restauración facial GFPGAN (con fondo ESRGAN opcional) ─────────────────
class _ESRGANWrapper:
    """Wrapper para usar spandrel ESRGAN como bg_upsampler de facexlib."""
    def __init__(self, model, tile, job_id=""):
        self.model, self.tile, self.job_id = model, tile, job_id

    def enhance(self, img, outscale=None):
        import cv2
        # No aplica cap aquí — la entrada ya fue limitada antes de este punto
        t   = bgr_to_tensor(img)
        out = tensor_to_bgr(tile_process(t, self.model, self.tile,
                                         job_id=self.job_id))
        if outscale and outscale != self.model.scale:
            h = int(img.shape[0] * outscale)
            w = int(img.shape[1] * outscale)
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out, None


def run_face_restoration(job_id: str, input_path: Path, output_path: Path,
                         outscale: int, tile: int, fidelity: float, bg_enhance: bool):
    import cv2, torch
    import numpy as np

    device = torch.device(INFER_DEVICE)

    # ── Cargar GFPGAN ────────────────────────────────────────────────────────
    jlog(job_id, "info", "Cargando GFPGAN (spandrel)…")
    gfp  = load_spandrel(MODELS_DIR / "GFPGANv1.4.pth")
    net = gfp.model.to(device)

    # ── Background upsampler ─────────────────────────────────────────────────
    bg_up = None
    if bg_enhance and (MODELS_DIR / "RealESRGAN_x4plus.pth").exists():
        try:
            jlog(job_id, "info", "Cargando ESRGAN para fondo…")
            esrgan = load_spandrel(MODELS_DIR / "RealESRGAN_x4plus.pth")
            bg_up  = _ESRGANWrapper(esrgan, tile, job_id)
        except Exception as e:
            jlog(job_id, "warn", f"No se pudo cargar ESRGAN para fondo: {e}")
    else:
        if not bg_enhance:
            jlog(job_id, "info", "Mejora de fondo desactivada (Optimización rápida).")

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer: {input_path.name}")

    # ── Cap resolución de salida ─────────────────────────────────────────────
    img, outscale = _prepare_input(job_id, img, outscale)

    # FaceRestoreHelper debe usar la escala efectiva después del límite 4K.
    jlog(job_id, "info", "Iniciando FaceRestoreHelper (facexlib)…")
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    face_helper = FaceRestoreHelper(
        upscale_factor=outscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )

    h_in, w_in = img.shape[:2]
    jlog(job_id, "info", f"Imagen (tras cap): {w_in}×{h_in}px → salida ≈ {w_in*outscale}×{h_in*outscale}px")

    # Upscale fondo
    if bg_up:
        jlog(job_id, "info", "Upscaleando fondo con ESRGAN…")
        bg_img, _ = bg_up.enhance(img, outscale=outscale)
    else:
        if outscale != 1:
            jlog(job_id, "info", f"Redimensionando fondo ({outscale}x) sin IA para ahorrar tiempo…")
            bg_img = cv2.resize(img, (int(w_in*outscale), int(h_in*outscale)), interpolation=cv2.INTER_LANCZOS4)
        else:
            bg_img = None

    face_helper.clean_all()
    face_helper.read_image(img)
    check_cancel(job_id)
    jlog(job_id, "info", "Detectando rostros…")
    face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    face_helper.align_warp_face()
    n_faces = len(face_helper.cropped_faces)
    jlog(job_id, "info", f"Rostros detectados: {n_faces}")

    if n_faces == 0:
        jlog(job_id, "warn", "Sin rostros detectados. Guardando fondo mejorado.")
        if bg_img is not None:
            _imwrite(bg_img, output_path)
        else:
            _imwrite(img, output_path)
        return

    for i, cropped in enumerate(face_helper.cropped_faces):
        check_cancel(job_id)
        jlog(job_id, "info", f"Restaurando rostro {i+1}/{n_faces} (fidelidad={fidelity})…")
        face_t = bgr_to_tensor(cropped, device=device)
        with torch.no_grad():
            result = net(face_t)
            out_face = result[0] if isinstance(result, (list, tuple)) else result
        
        alpha = 1.0 - fidelity # 0.1 fidelity (reconstruct) -> alpha = 0.9. 1.0 fidelity (original) -> alpha = 0.0
        restored_face = tensor_to_bgr(out_face)
        if alpha < 1.0:
            cropped_resized = cv2.resize(cropped, (restored_face.shape[1], restored_face.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            restored_face = cv2.addWeighted(restored_face, alpha, cropped_resized, 1.0 - alpha, 0)
            
        face_helper.add_restored_face(restored_face)

    face_helper.get_inverse_affine(None)
    restored = face_helper.paste_faces_to_input_image(upsample_img=bg_img)

    if restored is None or restored.size == 0:
        raise RuntimeError("paste_faces_to_input_image devolvió imagen vacía")

    jlog(job_id, "info", f"Guardando → {output_path.name}")
    _imwrite(restored, output_path)
    h_out, w_out = restored.shape[:2]
    jlog(job_id, "ok", f"Restauración completada: {w_in}×{h_in} → {w_out}×{h_out}")


# ─── PRO Edition (DAT + SkinDetail) ──────────────────────────────────────────
def run_pro_workflow(job_id: str, input_path: Path, output_path: Path,
                     outscale: int, tile: int, sharpen_alpha: float):
    import cv2
    jlog(job_id, "info", "Iniciando Workflow PRO 2026…")
    
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {input_path.name}")

    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        alpha = img[:, :, 3]
        img = img[:, :, :3]
        
    img, outscale_eff = _prepare_input(job_id, img, outscale)
    h_in, w_in = img.shape[:2]
    
    # Etapa 1: 4x FaceUpDAT
    model_dat_path = MODELS_DIR / "4xFaceUpDAT.pth"
    if not model_dat_path.exists():
        jlog(job_id, "warn", "4xFaceUpDAT no encontrado, usando RealESRGAN_x4plus como fallback.")
        model_dat_path = MODELS_DIR / "RealESRGAN_x4plus.pth"
    
    jlog(job_id, "info", f"Etapa 1: Upscale 4x con {model_dat_path.name}…")
    model_dat = load_spandrel(model_dat_path)
    check_cancel(job_id)
    img_t = bgr_to_tensor(img)
    out_1_t = tile_process(img_t, model_dat, tile, job_id=job_id)
    out_1 = tensor_to_bgr(out_1_t)
    
    # Etapa 2: 1x Skin Detail
    model_skin_path = MODELS_DIR / "1x-ITF-SkinDiffDetail-Lite-v1.pth"
    if model_skin_path.exists():
        jlog(job_id, "info", "Etapa 2: Upscale 1x de detalles de piel…")
        model_skin = load_spandrel(model_skin_path)
        check_cancel(job_id)
        out_1_t = bgr_to_tensor(out_1)
        out_2_t = tile_process(out_1_t, model_skin, tile, job_id=job_id)
        out_2 = tensor_to_bgr(out_2_t)
    else:
        jlog(job_id, "warn", "1x-ITF-SkinDiffDetail-Lite-v1 no encontrado, saltando etapa 2.")
        out_2 = out_1
        
    # Re-escala fina (Lanczos). Usa la escala efectiva calculada por el límite 4K.
    if outscale_eff != 4:
        target_w = int(w_in * outscale_eff)
        target_h = int(h_in * outscale_eff)
        jlog(job_id, "info", f"Re-escalando de 4x a {outscale_eff}x ({target_w}×{target_h}px)…")
        out_2 = cv2.resize(out_2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
    # Sharpen
    if sharpen_alpha > 0:
        jlog(job_id, "info", f"Aplicando Sharpen (alpha={sharpen_alpha})…")
        blur = cv2.GaussianBlur(out_2, (0, 0), sigmaX=0.85)
        out_2 = cv2.addWeighted(out_2, 1.0 + sharpen_alpha, blur, -sharpen_alpha, 0)
        
    if has_alpha:
        alpha_up = cv2.resize(alpha, (out_2.shape[1], out_2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        import numpy as np
        out_2 = np.dstack([out_2, alpha_up])

    jlog(job_id, "info", f"Guardando → {output_path.name}")
    _imwrite(out_2, output_path)
    h_out, w_out = out_2.shape[:2]
    jlog(job_id, "ok", f"PRO completado: {w_in}×{h_in} → {w_out}×{h_out}")


# ─── Dispatcher ──────────────────────────────────────────────────────────────
def dispatch(job_id: str, input_path: Path, output_path: Path,
             outscale: int, tile: int, fidelity: float, bg_enhance: bool,
             workflow: str, sharpen_alpha: float):
    try:
        if workflow == "pro":
            run_pro_workflow(job_id, input_path, output_path, outscale, tile, sharpen_alpha)
        else:
            run_face_restoration(job_id, input_path, output_path, outscale, tile, fidelity, bg_enhance)
    except Exception as e:
        jlog(job_id, "error", f"Restauración falló: {e}")
        raise e


# ─── Job runner ───────────────────────────────────────────────────────────────
def _process_one(job_id: str, file_id: str,
                 outscale: int, out_fmt: str, tile: int, fidelity: float, bg_enhance: bool,
                 workflow: str, sharpen_alpha: float):
    finfo = uploaded_files.get(file_id)
    if not finfo:
        return
    
    inp = Path(finfo["path"])
    ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(out_fmt, ".jpg")
    stem     = Path(finfo["original_name"]).stem
    out_name = f"{stem}_up{outscale}x{ext}"
    out_path = OUTPUT_DIR / job_id / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jobs_lock:
        for f in jobs[job_id]["files"]:
            if f["file_id"] == file_id:
                f["status"] = "processing"; f["started_at"] = time.time()
                break

    jlog(job_id, "info", f"── Inicio: {finfo['original_name']} [{workflow.upper()}] {outscale}x ──")
    t0 = time.time()

    try:
        from PIL import Image
        with Image.open(inp) as im:
            orig_w, orig_h = im.size

        dispatch(job_id, inp, out_path, outscale, tile, fidelity, bg_enhance, workflow, sharpen_alpha)

        with Image.open(out_path) as im:
            out_w, out_h = im.size

        elapsed = round(time.time() - t0, 1)
        with jobs_lock:
            for f in jobs[job_id]["files"]:
                if f["file_id"] == file_id:
                    f.update(status="completed", output_name=out_name,
                             output_path=str(out_path),
                             orig_size=[orig_w, orig_h],
                             out_size=[out_w, out_h], elapsed=elapsed)
                    break
            jobs[job_id]["completed"] += 1

        make_thumb(out_path, THUMBS_DIR / f"{job_id}_{file_id}_out.jpg")
        jlog(job_id, "ok", f"✓ {finfo['original_name']} — {elapsed}s")

    except JobCancelledException as e:
        jlog(job_id, "warn", f"Cancelado: {finfo['original_name']}")
        with jobs_lock:
            for f in jobs[job_id]["files"]:
                if f["file_id"] == file_id:
                    f["status"] = "cancelled"
                    f["error"]  = str(e)
                    break

    except Exception as e:
        tb = traceback.format_exc()
        jlog(job_id, "error", f"FALLO en {finfo['original_name']}: {e}")
        jlog(job_id, "error", tb.strip().split("\n")[-1])
        with jobs_lock:
            for f in jobs[job_id]["files"]:
                if f["file_id"] == file_id:
                    f["status"] = "error"
                    f["error"]  = str(e)
                    f["traceback"] = tb
                    break
            jobs[job_id]["errors"] += 1


def run_job(job_id: str):
    with jobs_lock:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = time.time()
    job = jobs[job_id]
    for finfo in job["files"]:
        with jobs_lock:
            if jobs[job_id].get("cancelled"):
                for f in jobs[job_id]["files"]:
                    if f["status"] == "queued":
                        f["status"] = "cancelled"
                        f["error"] = "Proceso cancelado por el usuario"
                break

        _process_one(job_id, finfo["file_id"],
                     job["outscale"], job["out_fmt"], job["tile"],
                     job["fidelity"], job["bg_enhance"],
                     job.get("workflow", "gfpgan"), job.get("sharpen_alpha", 0.25))

    # ── Limpiar directorio vacío si ningún archivo se procesó ────────────────
    job_out_dir = OUTPUT_DIR / job_id
    if job_out_dir.exists():
        try:
            contents = list(job_out_dir.iterdir())
            if not contents:
                job_out_dir.rmdir()
                jlog(job_id, "info", "Directorio de salida vacío → eliminado automáticamente.")
        except Exception:
            pass

    with jobs_lock:
        jobs[job_id]["status"]      = "completed"
        jobs[job_id]["finished_at"] = time.time()
    jlog(job_id, "ok",
         f"Job completado — {jobs[job_id]['completed']} OK / {jobs[job_id]['errors']} errores")


# ─── Gemini Analysis ──────────────────────────────────────────────────────────
GEMINI_PROMPT = """Analyze this portrait/model photography and return ONLY a JSON object (no markdown, no explanation):
{
  "has_face": boolean,
  "face_count": number,
  "face_coverage_pct": number,
  "is_portrait": boolean,
  "is_body_shot": boolean,
  "image_quality": "low"|"medium"|"high",
  "blur_detected": boolean,
  "recommended_fidelity": number,
  "bg_enhance": boolean,
  "recommended_scale": 2|4,
  "notes": string
}
Rules:
- The app is optimized purely for restoring human faces.
- recommended_fidelity (0.1 to 1.0): Use 0.1 to 0.4 for highly blurry/low-res faces (requires heavy AI reconstruction). Use 0.5 to 0.7 for normal restoration. Use 0.8 to 1.0 for high-quality photos where you must preserve the exact facial identity (no "wax" effect).
- bg_enhance (boolean): Set to true ONLY if the background contains important sharp details. If the background is blurry (bokeh), solid color, or irrelevant, set to false to save processing time!
- Blurry or low-res → recommended_scale: 4
- Already high-res → recommended_scale: 2
- notes: 1 concise sentence in Spanish explaining the fidelity and background decision."""


@app.post("/api/analyze")
async def api_analyze(request: Request):
    global _gemini_key
    data    = await request.json()
    file_id = data.get("file_id")
    api_key = data.get("api_key", _gemini_key).strip()

    if not api_key:
        raise HTTPException(400, "API key de Gemini no configurada")
    if file_id not in uploaded_files:
        raise HTTPException(404, "Archivo no encontrado")

    _gemini_key = api_key  # Persistir en sesión

    finfo = uploaded_files[file_id]
    try:
        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=api_key)

        # Intentar modelos en orden de preferencia (2025-2026)
        model_names = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash-latest",
        ]
        resp = None
        img  = Image.open(finfo["path"])
        last_err = None
        for mname in model_names:
            try:
                gmodel = genai.GenerativeModel(mname)
                resp   = gmodel.generate_content([GEMINI_PROMPT, img])
                print(f"[gemini] Usando modelo: {mname}")
                break
            except Exception as e:
                last_err = e
                print(f"[gemini] {mname} no disponible: {e}")
                continue

        if resp is None:
            raise RuntimeError(f"Ningún modelo Gemini disponible. Último error: {last_err}")

        text = resp.text.strip()
        # Limpiar posibles bloques markdown
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Error Gemini: {e}")


@app.post("/api/set-gemini-key")
async def api_set_key(request: Request):
    global _gemini_key
    data = await request.json()
    _gemini_key = data.get("key", "").strip()
    return JSONResponse({"ok": bool(_gemini_key)})


# ─── REST endpoints ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

@app.get("/api/info")
async def api_info():
    return {"device": DEVICE, "models": check_models(),
            "version": "3.0.0", "gemini_ready": bool(_gemini_key)}

@app.post("/api/upload")
async def api_upload(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        fid     = str(uuid.uuid4())
        dst     = UPLOAD_DIR / f"{fid}{ext}"
        content = await file.read()
        dst.write_bytes(content)
        try:
            from PIL import Image
            with Image.open(dst) as im: w, h = im.size
        except Exception: w, h = 0, 0
        uploaded_files[fid] = {"file_id": fid, "original_name": file.filename,
                                "path": str(dst), "size": len(content),
                                "dimensions": [w, h]}
        make_thumb(dst, THUMBS_DIR / f"{fid}.jpg")
        results.append({"file_id": fid, "original_name": file.filename,
                         "size": len(content), "dimensions": [w, h]})
    return JSONResponse({"files": results})

@app.get("/api/thumb/{file_id}")
async def api_thumb(file_id: str):
    p = THUMBS_DIR / f"{file_id}.jpg"
    if p.exists(): return FileResponse(str(p), media_type="image/jpeg")
    raise HTTPException(404)

@app.get("/api/thumb/{job_id}/{file_id}/out")
async def api_thumb_out(job_id: str, file_id: str):
    p = THUMBS_DIR / f"{job_id}_{file_id}_out.jpg"
    if p.exists(): return FileResponse(str(p), media_type="image/jpeg")
    raise HTTPException(404)

@app.get("/api/preview/{file_id}")
async def api_preview(file_id: str):
    fi = uploaded_files.get(file_id)
    if fi:
        p = Path(fi["path"])
        if p.exists(): return FileResponse(str(p))
    raise HTTPException(404)

@app.post("/api/process")
async def api_process(request: Request, background_tasks: BackgroundTasks):
    data     = await request.json()
    file_ids = data.get("file_ids", [])
    outscale = int(data.get("outscale", 4))
    out_fmt  = data.get("out_fmt",  "jpeg")
    tile     = int(data.get("tile", 512))
    fidelity = float(data.get("fidelity", 0.5))
    bg_enhance = bool(data.get("bg_enhance", False))
    workflow = data.get("workflow", "gfpgan")
    sharpen_alpha = float(data.get("sharpen_alpha", 0.25))

    if not file_ids:
        raise HTTPException(400, "Sin archivos")
    if outscale not in (2, 4):
        raise HTTPException(400, "Escala no válida: usa 2x o 4x")
    if out_fmt not in {"jpeg", "png", "webp"}:
        raise HTTPException(400, "Formato de salida no válido")
    if tile not in {128, 256, 384, 512, 640, 768, 896, 1024}:
        raise HTTPException(400, "Tile size no válido")
    if workflow not in {"gfpgan", "pro"}:
        raise HTTPException(400, "Workflow no válido")
    if not 0.1 <= fidelity <= 1.0:
        raise HTTPException(400, "Fidelidad fuera de rango")
    if not 0.0 <= sharpen_alpha <= 0.45:
        raise HTTPException(400, "Sharpen fuera de rango")
    m = check_models()
    if workflow == "gfpgan" and not m.get("gfpgan"):
        raise HTTPException(400, "Modelo GFPGAN no descargado")
    if workflow == "pro" and not m.get("pro_dat"):
        raise HTTPException(400, "Modelo 4xFaceUpDAT no descargado")

    job_id = str(uuid.uuid4())
    files_list = [
        {"file_id": fid,
         "original_name": uploaded_files[fid]["original_name"],
         "status": "queued", "output_name": None, "output_path": None,
         "error": None, "traceback": None,
         "orig_size": uploaded_files[fid]["dimensions"],
         "out_size": None, "elapsed": None}
        for fid in file_ids if fid in uploaded_files
    ]
    if not files_list:
        raise HTTPException(400, "Ninguno de los archivos solicitados existe en esta sesión")

    jobs[job_id] = {
        "job_id": job_id, "status": "queued",
        "total": len(files_list), "completed": 0, "errors": 0,
        "outscale": outscale, "out_fmt": out_fmt, "tile": tile,
        "fidelity": fidelity, "bg_enhance": bg_enhance,
        "workflow": workflow, "sharpen_alpha": sharpen_alpha,
        "files": files_list, "logs": [], "cancelled": False,
        "created_at": time.time(), "started_at": None, "finished_at": None
    }
    background_tasks.add_task(run_job, job_id)
    return JSONResponse({"job_id": job_id})

@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    job = jobs.get(job_id)
    if not job: raise HTTPException(404)
    return JSONResponse(job)

@app.get("/api/download/{job_id}/{file_id}")
async def api_dl_one(job_id: str, file_id: str):
    job = jobs.get(job_id)
    if not job: raise HTTPException(404)
    for f in job["files"]:
        if f["file_id"] == file_id and f["output_path"]:
            p = Path(f["output_path"])
            if p.exists():
                return FileResponse(str(p), filename=f["output_name"],
                                    media_type="application/octet-stream")
    raise HTTPException(404, "Archivo no procesado")

@app.get("/api/download-all/{job_id}")
async def api_dl_all(job_id: str):
    job = jobs.get(job_id)
    if not job: raise HTTPException(404)
    zp = OUTPUT_DIR / f"{job_id}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in job["files"]:
            if f["output_path"] and Path(f["output_path"]).exists():
                zf.write(f["output_path"], f["output_name"])
    return FileResponse(str(zp), filename=f"upscaled_{job_id[:8]}.zip",
                        media_type="application/zip")

@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["cancelled"] = True
            return JSONResponse({"ok": True})
    raise HTTPException(404)

@app.delete("/api/jobs/{job_id}")
async def api_del_job(job_id: str):
    if job_id not in jobs: raise HTTPException(404)
    
    # 1. Eliminar carpeta de resultados
    shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
    (OUTPUT_DIR / f"{job_id}.zip").unlink(missing_ok=True)
    
    # 2. Eliminar thumbnails específicos del job (los de salida)
    for f in jobs[job_id]["files"]:
        fid = f["file_id"]
        (THUMBS_DIR / f"{job_id}_{fid}_out.jpg").unlink(missing_ok=True)
        # Nota: Los thumbnails de entrada (fid.jpg) se borrarán por tiempo
        # ya que podrían ser compartidos si se implementara caché de subida.
    
    del jobs[job_id]
    return {"ok": True}


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # ── Limpiar directorios vacíos y archivos temporales al arrancar ──────────
    cleanup_old_files(max_age_hours=6) # Limpieza más agresiva al inicio
    start_housekeeper()

    m = check_models()
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ProUpscaler v3.0  ·  Spandrel + Gemini    ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  🐍 Python     : {sys.version.split()[0]}")
    print(f"  🖥  Device    : {DEVICE.upper()}")
    print(f"  🤖 GFPGAN     : {'✅' if m.get('gfpgan') else '❌  ejecuta download_models.py'}")
    print(f"  🤖 PRO DAT    : {'✅' if m.get('pro_dat') else '❌  faltan pesos en models/'}")
    print(f"  🤖 PRO Skin   : {'✅' if m.get('pro_skin') else '❌  (opcional) faltan pesos'}")
    print(f"  🤖 ESRGAN 4x  : {'✅' if m.get('realesrgan_x4') else '❌  ejecuta download_models.py'}")
    host = os.environ.get("PROUPSCALER_HOST", "127.0.0.1")
    port = int(os.environ.get("PROUPSCALER_PORT", "8765"))
    display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    print("────────────────────────────────────────────────────────────────")
    print(f"\n  🌐 http://{display_host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
