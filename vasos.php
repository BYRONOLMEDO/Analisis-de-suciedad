# ============================================================
# COMPARADOR DE VASOS (VIDRIO) – Colab, “detalle extremo”
# ============================================================
# Qué hace:
#   1) Te pide subir VARIOS VASOS LIMPIOS (una sola acción).
#   2) Para CADA vaso limpio, te pide subir VARIOS VASOS SUCIOS asociados.
#   3) Calcula un PUNTAJE DE SUCIEDAD (0 = limpio, 100 = más sucio)
#      pensado para vidrio, mitigando reflejos y variaciones de luz.
#   4) Muestra por cada vaso limpio:
#        - Tabla ordenada de MÁS sucio a MÁS limpio (con nombres de archivo)
#        - Gráfico de barras
#        - CSV guardado en /content/resultados_vasos/
#      y además un resumen_general.csv con TODAS las comparaciones.
#
# Cómo calcula la suciedad en vidrio:
#   - Conversión a escala de grises + normalización local (suavizado Gaussiano)
#     para reducir efectos de iluminación/reflejos típicos del vidrio.
#   - Máscara ANULAR (corona circular): ignora el borde externo y el centro
#     extremo; se enfoca en la “zona útil” donde suelen quedar marcas.
#   - Métrica híbrida:
#       base  = media(|sucio - limpio|) en la máscara
#       foco  = media de los valores del percentil 95 (mancha concentrada)
#       score = (1 - conc_weight)*base + conc_weight*foco   → (0..100)
#
# Parámetros recomendados (puedes ajustarlos):
INNER_RATIO = 0.18    # radio interno relativo (anillo interno)  [0..0.9]
OUTER_RATIO = 0.48    # radio externo relativo (anillo externo)  [INNER..0.5..0.9]
CONC_WEIGHT = 0.35    # peso de “mancha concentrada” en el score [0..1]
GAUSS_SIGMA  = 7.0    # sigma del suavizado (mitiga variaciones de luz)
DEBUG_VISUAL = False  # poner True para ver ejemplo de máscara/normalización

import io, os, math, warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

#---------------------------------------
# Utilidades de imagen
#---------------------------------------
def _to_pil(img_bytes_or_pil):
    """Acepta bytes o PIL.Image y devuelve PIL.Image en RGB, corrigiendo EXIF."""
    if isinstance(img_bytes_or_pil, Image.Image):
        im = img_bytes_or_pil
    else:
        im = Image.open(io.BytesIO(img_bytes_or_pil))
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im

def _to_gray_np(img_pil):
    return np.asarray(ImageOps.grayscale(img_pil), dtype=np.float32)

def _resize_like(img_pil, ref_pil):
    if img_pil.size != ref_pil.size:
        return img_pil.resize(ref_pil.size, Image.BICUBIC)
    return img_pil

def _gauss_blur(arr, sigma=GAUSS_SIGMA):
    """Suavizado Gaussiano usando OpenCV si está, si no: aproximación por box filter."""
    try:
        import cv2
        ksize = 0  # auto a partir de sigma
        return cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
    except Exception:
        # Fallback simple: media móvil separable 3x3 varias veces
        ker = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float32) / 9.0
        out = arr.copy()
        for _ in range(3):
            out = _convolve2d_same(out, ker)
        return out

def _convolve2d_same(a, k):
    """Convolución 2D simple 'same' (lenta, pero sin dependencias)."""
    h, w = a.shape
    kh, kw = k.shape
    pad_y, pad_x = kh//2, kw//2
    ap = np.pad(a, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')
    out = np.zeros_like(a, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            region = ap[y:y+kh, x:x+kw]
            out[y, x] = float((region * k).sum())
    return out

def _local_norm(gray_np, sigma=GAUSS_SIGMA):
    """Normaliza iluminación: resta fondo suavizado y re-centra."""
    bg = _gauss_blur(gray_np, sigma=sigma)
    enh = gray_np - bg + 128.0
    return np.clip(enh, 0, 255)

def _annulus_mask(h, w, inner_ratio=INNER_RATIO, outer_ratio=OUTER_RATIO):
    """Máscara anular (corona circular) centrada en la imagen."""
    cy, cx = h//2, w//2
    r_in = int(max(0, min(1, inner_ratio)) * min(h, w))
    r_out = int(max(r_in+1, min(1, outer_ratio)) * min(h, w))
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy)**2 + (xx - cx)**2
    return (dist2 >= r_in*r_in) & (dist2 <= r_out*r_out)

#---------------------------------------
# Métrica de suciedad (vidrio)
#---------------------------------------
def dirtiness_score_glass(clean_pil, dirty_pil,
                          inner_ratio=INNER_RATIO,
                          outer_ratio=OUTER_RATIO,
                          conc_weight=CONC_WEIGHT,
                          sigma=GAUSS_SIGMA):
    """Devuelve (score_total, base_media, foco_p95, p95, mean, std)."""
    dirty_pil = _resize_like(dirty_pil, clean_pil)
    gC = _to_gray_np(clean_pil)
    gD = _to_gray_np(dirty_pil)

    gC = _local_norm(gC, sigma=sigma)
    gD = _local_norm(gD, sigma=sigma)

    diff = np.abs(gD - gC)  # 0..255
    h, w = diff.shape
    mask = _annulus_mask(h, w, inner_ratio, outer_ratio)
    if not mask.any():
        m = diff.mean()
        base = (m/255.0)*100.0
        return base, base, base, np.percentile(diff,95), m, diff.std()

    region = diff[mask]
    mean = region.mean()
    std  = region.std()
    p95  = np.percentile(region, 95)

    base_media = (mean/255.0)*100.0
    foco_p95   = (region[region >= p95].mean()/255.0)*100.0 if np.any(region >= p95) else base_media

    score = (1.0 - conc_weight)*base_media + conc_weight*foco_p95
    return float(score), float(base_media), float(foco_p95), float(p95), float(mean), float(std)

#---------------------------------------
# Flujo de Colab: subidas y análisis
#---------------------------------------
from google.colab import files
from IPython.display import display

print("Parámetros de análisis (puedes modificarlos arriba):")
print(f"  INNER_RATIO={INNER_RATIO}, OUTER_RATIO={OUTER_RATIO}, CONC_WEIGHT={CONC_WEIGHT}, GAUSS_SIGMA={GAUSS_SIGMA}")
print("------------------------------------------------------")

# 1) Subir VASOS LIMPIOS
print("📤 Sube AHORA tus VASOS LIMPIOS (puedes seleccionar varios a la vez).")
uploaded_clean = files.upload()

clean_images = {}
for name, data in uploaded_clean.items():
    clean_images[name] = _to_pil(data)

if not clean_images:
    raise SystemExit("No se cargaron vasos limpios. Vuelve a ejecutar y súbelos.")

print(f"✅ Vasos limpios cargados: {list(clean_images.keys())}")

# 2) Subir VASOS SUCIOS POR CADA LIMPIO
sucios_por_limpio = {}
for clean_name in clean_images.keys():
    print(f"\n📤 Sube AHORA los VASOS SUCIOS para: {clean_name} (puedes seleccionar varios).")
    up_dirty = files.upload()
    sucios_por_limpio[clean_name] = {dname: _to_pil(ddata) for dname, ddata in up_dirty.items()}
    print(f"   ➜ Sucios asociados: {list(sucios_por_limpio[clean_name].keys())}")

# Carpeta de salida
os.makedirs("resultados_vasos", exist_ok=True)

# 3) ANÁLISIS
resumen_rows = []
first_pair_visual_done = False

for clean_name, dirty_dict in sucios_por_limpio.items():
    clean_img = clean_images[clean_name]

    resultados = []
    for dname, dimg in dirty_dict.items():
        score, base, foco, p95, mean, std = dirtiness_score_glass(
            clean_img, dimg,
            inner_ratio=INNER_RATIO,
            outer_ratio=OUTER_RATIO,
            conc_weight=CONC_WEIGHT,
            sigma=GAUSS_SIGMA
        )
        resultados.append({
            "Vaso Limpio": clean_name,
            "Vaso Sucio": dname,
            "Nivel de Suciedad (%)": round(score, 2),
            "Base_media(%)": round(base, 2),
            "Foco_p95(%)": round(foco, 2),
            "p95(diff)": round(p95, 2),
            "mean(diff)": round(mean, 2),
            "std(diff)": round(std, 2),
        })

        # Visual debug (opcional, solo 1 par para no saturar)
        if DEBUG_VISUAL and not first_pair_visual_done:
            import matplotlib.pyplot as plt
            gC = _local_norm(_to_gray_np(_resize_like(clean_img, clean_img)), sigma=GAUSS_SIGMA)
            gD = _local_norm(_to_gray_np(_resize_like(dimg, clean_img)),  sigma=GAUSS_SIGMA)
            diff = np.abs(gD - gC)
            mask = _annulus_mask(*diff.shape, INNER_RATIO, OUTER_RATIO)
            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1); plt.imshow(gC, cmap='gray'); plt.title('Limpio (norm.)'); plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(gD, cmap='gray'); plt.title('Sucio (norm.)');  plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(diff, cmap='gray'); plt.title('Diff + máscara'); plt.axis('off')
            # Pintar la máscara anular sobre la diff
            overlay = diff.copy()
            overlay[~mask] = overlay[~mask]*0.3
            plt.imshow(overlay, cmap='gray')
            plt.show()
            first_pair_visual_done = True

    if not resultados:
        print(f"\n⚠️ No se subieron sucios para {clean_name}.")
        continue

    # Ordenar de MÁS SUCIO → MÁS LIMPIO
    df = pd.DataFrame(resultados).sort_values(
        by="Nivel de Suciedad (%)", ascending=False
    ).reset_index(drop=True)

    # Mostrar en pantalla
    print(f"\n================== RESULTADOS para {clean_name} ==================")
    display(df)

    # También mostrar SOLO los nombres en orden (como pide el enunciado)
    print("\nOrden (de más sucio a más limpio):")
    for i, row in df.iterrows():
        print(f"  {i+1:02d}. {row['Vaso Sucio']}  →  {row['Nivel de Suciedad (%)']}%")

    # Gráfico de barras
    plt.figure(figsize=(9, 4))
    plt.title(f"Suciedad por vaso sucio vs {clean_name}")
    plt.bar(df["Vaso Sucio"], df["Nivel de Suciedad (%)"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Nivel de Suciedad (%)")
    plt.xlabel("Vasos Sucios")
    plt.tight_layout()
    plt.show()

    # Guardar CSV por vaso limpio
    base_clean = os.path.splitext(os.path.basename(clean_name))[0]
    out_csv = os.path.join("resultados_vasos", f"resultados_{base_clean}.csv")
    df.to_csv(out_csv, index=False)
    print(f"💾 Guardado CSV: {out_csv}")

    # Acumular para resumen general
    resumen_rows.extend(df.to_dict(orient="records"))

# 4) RESUMEN GENERAL
if resumen_rows:
    resumen = pd.DataFrame(resumen_rows).sort_values(
        by=["Vaso Limpio", "Nivel de Suciedad (%)"], ascending=[True, False]
    ).reset_index(drop=True)
    print("\n================== RESUMEN GENERAL (todas las comparaciones) ==================")
    display(resumen)

    resumen_path = os.path.join("resultados_vasos", "resumen_general.csv")
    resumen.to_csv(resumen_path, index=False)
    print(f"📁 CSVs guardados en: /content/resultados_vasos/")
else:
    print("\nNo se encontraron comparaciones. ¿Subiste sucios para cada vaso limpio?")
