"""
app.py — Backend Flask para MG Real Estate · CDMX
Predicción de precio de inmuebles con RF, XGBoost y LightGBM

Estructura esperada en el mismo directorio:
  models/
    rf_model_CDMX.pkl          ← pipeline sklearn entrenado
    xgb_model_CDMX.pkl
    lgbm_model_CDMX.pkl
    rf_model_metadata.json     ← JSON generado por el notebook
    xgb_model_metadata.json
    lgbm_model_metadata.json
  static/
    index.html
"""

import os
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────────
# Configuración de logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constantes geográficas (idénticas a los notebooks)
# ──────────────────────────────────────────────
ZOCALO_LAT,  ZOCALO_LON  = 19.4326, -99.1332
POLANCO_LAT, POLANCO_LON = 19.4328, -99.1929

CDMX_LAT_RANGE = (19.05, 19.65)
CDMX_LON_RANGE = (-99.55, -98.90)

# ──────────────────────────────────────────────
# Catálogos válidos
# ──────────────────────────────────────────────
VALID_PROPERTY_TYPES = ["apartment", "house", "store", "PH"]

VALID_ALCALDIAS = [
    "AlvaroObregon", "Azcapotzalco", "BenitoJuarez", "Coyoacan",
    "CuajimalpaDeMorelos", "Cuauhtemoc", "GustavoAMadero", "Iztacalco",
    "Iztapalapa", "LaMagdalenaContreras", "MiguelHidalgo", "MilpaAlta",
    "Tlahuac", "Tlalpan", "VenustianoCarranza", "Xochimilco",
]

# ──────────────────────────────────────────────────────────────
# Mapa de normalización: clave del formulario → clave en el JSON
# Los notebooks guardaron algunos nombres abreviados en el dataset
# (p.ej. "Cuajimalpa" en lugar de "CuajimalpaDeMorelos").
# Este diccionario traduce en ambas direcciones al leer el JSON.
# ──────────────────────────────────────────────────────────────
ALCALDIA_NAME_MAP: dict[str, str] = {
    # formulario  →  clave en alcaldia_medians del JSON
    "CuajimalpaDeMorelos":  "Cuajimalpa",
    "LaMagdalenaContreras": "MagdalenaContreras",
    "MilpaAlta":            "MilpaAlta",   # mismo, explícito por claridad
}
# Inverso: JSON key → formulario key (se construye automáticamente)
_ALCALDIA_NAME_MAP_INV: dict[str, str] = {v: k for k, v in ALCALDIA_NAME_MAP.items()}


def _form_key_to_json_key(place: str) -> str:
    """Convierte la clave del formulario a la clave usada en el JSON de metadata."""
    return ALCALDIA_NAME_MAP.get(place, place)


def _json_key_to_form_key(place: str) -> str:
    """Convierte la clave del JSON de metadata a la clave del formulario."""
    return _ALCALDIA_NAME_MAP_INV.get(place, place)


# ──────────────────────────────────────────────────────────────
# Medianas por alcaldía — se sobreescribirán con valores reales
# del JSON al arrancar; estas son el último recurso.
# ──────────────────────────────────────────────────────────────
ALCALDIA_MEDIAN_FALLBACK: dict[str, float] = {
    "AlvaroObregon":          2_900_000,
    "Azcapotzalco":           1_343_818,
    "BenitoJuarez":           2_696_000,
    "Coyoacan":               2_049_333,
    "CuajimalpaDeMorelos":    7_040_695,
    "Cuauhtemoc":             1_522_586,
    "GustavoAMadero":         1_560_000,
    "Iztacalco":              1_029_394,
    "Iztapalapa":               918_571,
    "LaMagdalenaContreras":   3_800_000,
    "MiguelHidalgo":          6_410_000,
    "MilpaAlta":              1_625_000,
    "Tlahuac":                1_020_000,
    "Tlalpan":                2_950_000,
    "VenustianoCarranza":       934_940,
    "Xochimilco":             2_710_000,
}

# ──────────────────────────────────────────────────────────────
# Métricas por modelo — se sobreescribirán desde los JSONs.
# ──────────────────────────────────────────────────────────────
MODEL_DEFAULT_METRICS: dict[str, dict] = {
    "rf": {
        "algorithm": "Random Forest",
        "r2_cv":   0.850,
        "r2_test": 0.739,
        "mae":     1_296_670,
        "rmse":    1_800_000,
        "mape":    28.74,
        "note":    "Modelo optimizado con Optuna. Robusto y explicable.",
    },
    "xgb": {
        "algorithm": "XGBoost",
        "r2_cv":   0.858,
        "r2_test": 0.720,
        "mae":     780_000,
        "rmse":    1_480_000,
        "mape":    29.0,
        "note":    "Mejor R² en CV. Gradient boosting de alta precisión.",
    },
    "lgbm": {
        "algorithm": "LightGBM",
        "r2_cv":   0.839,
        "r2_test": 0.740,
        "mae":     760_000,
        "rmse":    1_420_000,
        "mape":    32.0,
        "note":    "Mejor R² en escala MXN. Entrenamiento ~5× más rápido.",
    },
}

# ──────────────────────────────────────────────
# Carga de modelos
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
STATIC_DIR  = BASE_DIR / "static"

MODEL_FILES = {
    "rf":   MODELS_DIR / "rf_model_CDMX.pkl",
    "xgb":  MODELS_DIR / "xgb_model_CDMX.pkl",
    "lgbm": MODELS_DIR / "lgbm_model_CDMX.pkl",
}

METADATA_FILES = {
    "rf":   MODELS_DIR / "rf_model_metadata.json",
    "xgb":  MODELS_DIR / "xgb_model_metadata.json",
    "lgbm": MODELS_DIR / "lgbm_model_metadata.json",
}

_models: dict[str, object] = {}
# Raw metadata dicts, accesibles desde los endpoints si se necesita
_metadata: dict[str, dict] = {}


def _load_metadata() -> None:
    """
    Lee los tres JSONs de metadata generados por los notebooks y:

    1. Sobreescribe ALCALDIA_MEDIAN_FALLBACK con las medianas reales
       del dataset (normalizando los nombres de alcaldía al esquema
       del formulario mediante ALCALDIA_NAME_MAP).

    2. Sobreescribe MODEL_DEFAULT_METRICS con las métricas exactas
       de cada modelo (R², MAE, RMSE, MAPE).

    3. Actualiza las constantes geográficas ZOCALO/POLANCO si el JSON
       las trae (por si cambian en una re-ejecución del notebook).

    Se llama una sola vez al arrancar, antes de recibir peticiones.
    """
    global ZOCALO_LAT, ZOCALO_LON, POLANCO_LAT, POLANCO_LON

    medians_loaded = False  # basta con leer las medianas de un solo JSON

    for key, path in METADATA_FILES.items():
        if not path.exists():
            logger.warning("⚠️  Metadata '%s' no encontrada en %s", key, path)
            continue

        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ Error leyendo metadata '%s': %s", key, exc)
            continue

        _metadata[key] = meta
        logger.info("✅ Metadata '%s' cargada desde %s", key, path)

        # ── 1. Medianas por alcaldía ──────────────────────────────
        # Solo necesitamos cargarlas una vez; las tres son iguales
        # porque provienen del mismo dataset.
        if not medians_loaded and "alcaldia_medians" in meta:
            raw_medians: dict = meta["alcaldia_medians"]
            for json_key, median_value in raw_medians.items():
                # Traducir al esquema del formulario
                form_key = _json_key_to_form_key(json_key)
                ALCALDIA_MEDIAN_FALLBACK[form_key] = float(median_value)
                # Guardar también con la clave original por si acaso
                if form_key != json_key:
                    ALCALDIA_MEDIAN_FALLBACK[json_key] = float(median_value)
            medians_loaded = True
            logger.info(
                "   → %d medianas de alcaldía cargadas desde '%s'",
                len(raw_medians), key,
            )

        # ── 2. Métricas del modelo ────────────────────────────────
        algo_names = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
        notes = {
            "rf":   "Optimizado con Optuna. Robusto y explicable.",
            "xgb":  "Mayor R² en validación cruzada. Gradient boosting de alta precisión.",
            "lgbm": "Mejor R² en escala MXN. Entrenamiento ~5× más rápido que RF.",
        }

        MODEL_DEFAULT_METRICS[key].update({
            "algorithm": algo_names.get(key, key),
            "r2_cv":     float(meta.get("cv_r2_mean",  MODEL_DEFAULT_METRICS[key]["r2_cv"])),
            "r2_std":    float(meta.get("cv_r2_std",   0.0)),
            "r2_test":   float(meta.get("test_r2_mxn", MODEL_DEFAULT_METRICS[key]["r2_test"])),
            "r2_log":    float(meta.get("test_r2_log", 0.0)),
            "mae":       float(meta.get("test_mae",    MODEL_DEFAULT_METRICS[key]["mae"])),
            "mape":      float(meta.get("test_mape",   MODEL_DEFAULT_METRICS[key]["mape"])),
            # RMSE no está en el JSON; lo aproximamos como 1.35 × MAE si no existe
            "rmse":      float(meta.get("test_rmse",
                               meta.get("test_mae", MODEL_DEFAULT_METRICS[key]["mae"]) * 1.35)),
            "note":      notes.get(key, ""),
        })
        logger.info(
            "   → Métricas '%s': R²CV=%.4f | MAE=$%,.0f | MAPE=%.1f%%",
            key,
            MODEL_DEFAULT_METRICS[key]["r2_cv"],
            MODEL_DEFAULT_METRICS[key]["mae"],
            MODEL_DEFAULT_METRICS[key]["mape"],
        )

        # ── 3. Coordenadas de referencia (opcional) ───────────────
        if "zocalo" in meta:
            ZOCALO_LAT = float(meta["zocalo"].get("lat", ZOCALO_LAT))
            ZOCALO_LON = float(meta["zocalo"].get("lon", ZOCALO_LON))
        if "polanco" in meta:
            POLANCO_LAT = float(meta["polanco"].get("lat", POLANCO_LAT))
            POLANCO_LON = float(meta["polanco"].get("lon", POLANCO_LON))


def _load_models() -> None:
    """Carga los tres modelos .pkl al arrancar la aplicación."""
    for key, path in MODEL_FILES.items():
        if path.exists():
            try:
                _models[key] = joblib.load(path)
                logger.info("✅ Modelo '%s' cargado desde %s", key, path)
            except Exception as exc:  # noqa: BLE001
                logger.error("❌ Error cargando modelo '%s': %s", key, exc)
        else:
            logger.warning("⚠️  Modelo '%s' no encontrado en %s", key, path)


# ──────────────────────────────────────────────
# Ingeniería de características
# (idéntica a la función construir_features de los notebooks)
# ──────────────────────────────────────────────

def _build_features(
    property_type: str,
    place: str,
    surface_total_m2: float,
    surface_covered_m2: float,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Construye el DataFrame de una fila con los 16 features del modelo.
    Aplica la misma lógica de los notebooks:
      - swap si cubierta > total
      - ratio_covered, log features
      - distancias al Zócalo y Polanco
      - zonas NS / EO
      - interacciones surf×tipo
      - place_median_price (fallback si alcaldía desconocida)
    """
    # Swap si superficie cubierta > total
    if surface_covered_m2 > surface_total_m2:
        surface_total_m2, surface_covered_m2 = surface_covered_m2, surface_total_m2

    place_med = ALCALDIA_MEDIAN_FALLBACK.get(
        place, float(np.median(list(ALCALDIA_MEDIAN_FALLBACK.values())))
    )

    dist_zocalo  = math.sqrt(
        ((lat - ZOCALO_LAT)  * 111.0) ** 2 +
        ((lon - ZOCALO_LON)  * 98.5)  ** 2
    )
    dist_polanco = math.sqrt(
        ((lat - POLANCO_LAT) * 111.0) ** 2 +
        ((lon - POLANCO_LON) * 98.5)  ** 2
    )

    row = {
        "property_type":         property_type,
        "places":                place,
        "surface_total_in_m2":   surface_total_m2,
        "surface_covered_in_m2": surface_covered_m2,
        "lat":                   lat,
        "lon":                   lon,
        "ratio_covered":         surface_covered_m2 / max(surface_total_m2, 1),
        "log_surface_total":     math.log1p(surface_total_m2),
        "log_surface_cov":       math.log1p(surface_covered_m2),
        "dist_zocalo_km":        dist_zocalo,
        "dist_polanco_km":       dist_polanco,
        "zone_ns":               int(lat > ZOCALO_LAT),
        "zone_eo":               int(lon > ZOCALO_LON),
        "surf_x_apt":            surface_total_m2 * int(property_type == "apartment"),
        "surf_x_house":          surface_total_m2 * int(property_type == "house"),
        "place_median_price":    place_med,
    }
    return pd.DataFrame([row])


# ──────────────────────────────────────────────
# Validación de entrada
# ──────────────────────────────────────────────

class ValidationError(ValueError):
    pass


def _validate_input(data: dict) -> dict:
    """
    Valida y coerciona los parámetros de la petición.
    Lanza ValidationError con mensaje descriptivo si algo falla.
    """
    errors: list[str] = []

    # --- property_type ---
    prop_type = data.get("property_type", "").strip()
    if prop_type not in VALID_PROPERTY_TYPES:
        errors.append(
            f"Tipo de propiedad inválido: '{prop_type}'. "
            f"Opciones: {VALID_PROPERTY_TYPES}"
        )

    # --- place (alcaldía) ---
    place = data.get("place", "").strip()
    if place not in VALID_ALCALDIAS:
        errors.append(
            f"Alcaldía inválida: '{place}'. "
            f"Opciones: {VALID_ALCALDIAS}"
        )

    # --- surface_total ---
    try:
        surface_total = float(data["surface_total"])
        if not (10 <= surface_total <= 5000):
            errors.append("Superficie total debe estar entre 10 y 5 000 m².")
    except (KeyError, TypeError, ValueError):
        errors.append("Superficie total es requerida y debe ser un número.")
        surface_total = 0.0

    # --- surface_covered ---
    try:
        surface_covered = float(data["surface_covered"])
        if not (10 <= surface_covered <= 5000):
            errors.append("Superficie cubierta debe estar entre 10 y 5 000 m².")
    except (KeyError, TypeError, ValueError):
        errors.append("Superficie cubierta es requerida y debe ser un número.")
        surface_covered = 0.0

    # --- lat ---
    try:
        lat = float(data["lat"])
        if not (CDMX_LAT_RANGE[0] <= lat <= CDMX_LAT_RANGE[1]):
            errors.append(
                f"Latitud fuera del rango CDMX "
                f"({CDMX_LAT_RANGE[0]}–{CDMX_LAT_RANGE[1]})."
            )
    except (KeyError, TypeError, ValueError):
        errors.append("Latitud es requerida y debe ser un número decimal.")
        lat = ZOCALO_LAT  # fallback

    # --- lon ---
    try:
        lon = float(data["lon"])
        if not (CDMX_LON_RANGE[0] <= lon <= CDMX_LON_RANGE[1]):
            errors.append(
                f"Longitud fuera del rango CDMX "
                f"({CDMX_LON_RANGE[0]}–{CDMX_LON_RANGE[1]})."
            )
    except (KeyError, TypeError, ValueError):
        errors.append("Longitud es requerida y debe ser un número decimal.")
        lon = ZOCALO_LON  # fallback

    # --- model ---
    model_key = data.get("model", "lgbm").strip().lower()
    if model_key not in MODEL_FILES:
        errors.append(
            f"Modelo inválido: '{model_key}'. "
            f"Opciones: {list(MODEL_FILES.keys())}"
        )

    if errors:
        raise ValidationError(" | ".join(errors))

    return {
        "property_type":    prop_type,
        "place":            place,
        "surface_total":    surface_total,
        "surface_covered":  surface_covered,
        "lat":              lat,
        "lon":              lon,
        "model_key":        model_key,
    }


# ──────────────────────────────────────────────
# Aplicación Flask
# ──────────────────────────────────────────────

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)  # Permite peticiones desde cualquier origen (útil en desarrollo)

# Orden importante: primero metadata (carga medianas y métricas reales
# desde los JSONs), luego los modelos .pkl.
_load_metadata()
_load_models()


# ── Servir el frontend ──────────────────────────────────────
@app.route("/")
def index():
    """Sirve el index.html desde la carpeta static/."""
    return send_from_directory(app.static_folder, "index.html")


# ── Catálogos ───────────────────────────────────────────────
@app.route("/api/catalogs", methods=["GET"])
def catalogs():
    """
    Devuelve los valores válidos para los selectores del formulario
    y las métricas de cada modelo.
    """
    return jsonify({
        "property_types": VALID_PROPERTY_TYPES,
        "alcaldias":       VALID_ALCALDIAS,
        "models":          MODEL_DEFAULT_METRICS,
        "loaded_models":   list(_models.keys()),
    })


# ── Predicción ──────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint principal de inferencia.

    Body JSON esperado:
    {
        "property_type":   "apartment",   // apartment | house | store | PH
        "place":           "BenitoJuarez",
        "surface_total":   120,
        "surface_covered": 110,
        "lat":             19.3682,
        "lon":             -99.1717,
        "model":           "lgbm"         // rf | xgb | lgbm
    }

    Respuesta JSON:
    {
        "success":         true,
        "predicted_price": 4200000,
        "model_key":       "lgbm",
        "algorithm":       "LightGBM",
        "metrics":         { r2_cv, r2_test, mae, rmse, mape, note },
        "zone_median":     3800000,
        "zone_name":       "BenitoJuarez",
        "inputs":          { ...parámetros limpios... }
    }
    """
    # 1. Parsear JSON
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"success": False, "error": "Body debe ser JSON válido."}), 400

    # 2. Validar
    try:
        params = _validate_input(data)
    except ValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422

    model_key = params["model_key"]

    # 3. Verificar que el modelo esté cargado
    if model_key not in _models:
        return jsonify({
            "success": False,
            "error":   (
                f"El modelo '{model_key}' no está disponible en este servidor. "
                "Asegúrese de que el archivo .pkl esté en la carpeta models/."
            ),
        }), 503

    # 4. Construir features
    try:
        X_new = _build_features(
            property_type      = params["property_type"],
            place              = params["place"],
            surface_total_m2   = params["surface_total"],
            surface_covered_m2 = params["surface_covered"],
            lat                = params["lat"],
            lon                = params["lon"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error construyendo features")
        return jsonify({"success": False, "error": f"Error interno al procesar datos: {exc}"}), 500

    # 5. Inferencia
    try:
        model = _models[model_key]
        log_pred = model.predict(X_new)[0]
        price_mxn = float(np.expm1(log_pred))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en la inferencia del modelo '%s'", model_key)
        return jsonify({"success": False, "error": f"Error en la predicción: {exc}"}), 500

    # 6. Datos de contexto de zona (para la gráfica)
    zone_median = ALCALDIA_MEDIAN_FALLBACK.get(params["place"], price_mxn)

    # 7. Métricas del modelo seleccionado
    metrics = MODEL_DEFAULT_METRICS.get(model_key, {})

    # 8. Log
    logger.info(
        "Predicción | modelo=%s | %s | %s | %.0fm² | lat=%.4f lon=%.4f | $%,.0f MXN",
        model_key,
        params["property_type"],
        params["place"],
        params["surface_total"],
        params["lat"],
        params["lon"],
        price_mxn,
    )

    return jsonify({
        "success":         True,
        "predicted_price": round(price_mxn, 2),
        "predicted_price_m2": round(price_mxn / max(params["surface_total"], 1), 2),
        "model_key":       model_key,
        "algorithm":       metrics.get("algorithm", model_key),
        "metrics":         metrics,
        "zone_median":     zone_median,
        "zone_name":       params["place"],
        "inputs": {
            "property_type":   params["property_type"],
            "place":           params["place"],
            "surface_total":   params["surface_total"],
            "surface_covered": params["surface_covered"],
            "lat":             params["lat"],
            "lon":             params["lon"],
        },
    })


# ── Comparar los 3 modelos de un solo golpe ─────────────────
@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Ejecuta la inferencia con los tres modelos en paralelo y devuelve
    un arreglo con los resultados para mostrar la comparativa.

    Mismo body JSON que /api/predict (sin el campo 'model').
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"success": False, "error": "Body debe ser JSON válido."}), 400

    results = []
    for key in MODEL_FILES:
        data["model"] = key
        try:
            params = _validate_input(data)
        except ValidationError as exc:
            results.append({"model_key": key, "error": str(exc)})
            continue

        if key not in _models:
            results.append({
                "model_key": key,
                "error": "Modelo no cargado.",
            })
            continue

        try:
            X_new     = _build_features(
                property_type      = params["property_type"],
                place              = params["place"],
                surface_total_m2   = params["surface_total"],
                surface_covered_m2 = params["surface_covered"],
                lat                = params["lat"],
                lon                = params["lon"],
            )
            log_pred  = _models[key].predict(X_new)[0]
            price_mxn = float(np.expm1(log_pred))
            metrics   = MODEL_DEFAULT_METRICS.get(key, {})
            results.append({
                "model_key":       key,
                "algorithm":       metrics.get("algorithm", key),
                "predicted_price": round(price_mxn, 2),
                "metrics":         metrics,
                "success":         True,
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error comparando modelo '%s'", key)
            results.append({"model_key": key, "error": str(exc)})

    zone_median = ALCALDIA_MEDIAN_FALLBACK.get(
        data.get("place", ""), 0.0
    )

    return jsonify({
        "success":     True,
        "results":     results,
        "zone_median": zone_median,
        "zone_name":   data.get("place", ""),
    })


# ── Health-check ────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "loaded_models":    list(_models.keys()),
        "missing_models":   [k for k in MODEL_FILES  if k not in _models],
        "loaded_metadata":  list(_metadata.keys()),
        "missing_metadata": [k for k in METADATA_FILES if k not in _metadata],
        "medians_source":   "json" if _metadata else "fallback",
        "alcaldia_medians": ALCALDIA_MEDIAN_FALLBACK,
    })


# ──────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("🚀 Iniciando MG Real Estate API en puerto %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
