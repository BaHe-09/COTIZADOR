"""Backend Flask para consumir modelos inmobiliarios CDMX.

Carga los tres modelos pre-entrenados (LGBM, Random Forest y XGBoost) y expone
endpoints JSON para que el cotizador HTML realice inferencia en tiempo real.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent

PROPERTY_TYPES = {
    "apartment": "Departamento",
    "house": "Casa",
    "store": "Local comercial",
    "PH": "Penthouse",
}

# Promedios aproximados por m² usados para comparar la predicción contra la zona.
ZONE_AVERAGE_M2 = {
    "BenitoJuarez": 42000,
    "Iztapalapa": 22000,
    "MiguelHidalgo": 72000,
    "Cuauhtemoc": 52000,
    "Coyoacan": 42000,
    "AlvaroObregon": 47000,
    "Azcapotzalco": 30000,
    "GustavoAMadero": 26000,
    "Iztacalco": 28000,
    "Tlalpan": 39000,
    "VenustianoCarranza": 27000,
    "Xochimilco": 24000,
    "Tlahuac": 21000,
    "MagdalenaContreras": 34000,
    "Cuajimalpa": 54000,
    "MilpaAlta": 18000,
}

# Métricas de validación de la actividad anterior. Si cuentas con un archivo
# metrics.json en la raíz, estos valores se reemplazan automáticamente.
DEFAULT_METRICS = {
    "lgbm": {"mae": 385000, "rmse": 690000, "r2": 0.88, "mape": 0.132},
    "rf": {"mae": 420000, "rmse": 760000, "r2": 0.85, "mape": 0.148},
    "xgb": {"mae": 405000, "rmse": 725000, "r2": 0.87, "mape": 0.139},
}

MODEL_FILES = {
    "lgbm": "lgbm_model_CDMX.pkl",
    "rf": "rf_model_CDMX.pkl",
    "xgb": "xgb_model_CDMX.pkl",
}

MODEL_LABELS = {
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


@dataclass
class LoadedModel:
    key: str
    label: str
    path: Path
    estimator: Any | None
    loaded: bool
    error: str | None = None


app = Flask(__name__)
_models: dict[str, LoadedModel] = {}
_metrics = DEFAULT_METRICS.copy()


def _is_lfs_pointer(path: Path) -> bool:
    """Detecta si el archivo .pkl es un pointer de Git LFS y no el binario real."""
    try:
        with path.open("rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def load_metrics() -> dict[str, dict[str, float]]:
    metrics_path = BASE_DIR / "metrics.json"
    if not metrics_path.exists():
        return DEFAULT_METRICS.copy()
    with metrics_path.open("r", encoding="utf-8") as fh:
        custom = json.load(fh)
    merged = DEFAULT_METRICS.copy()
    for key, values in custom.items():
        if key in merged:
            merged[key] = {**merged[key], **values}
    return merged


def load_models() -> dict[str, LoadedModel]:
    loaded: dict[str, LoadedModel] = {}
    for key, filename in MODEL_FILES.items():
        path = BASE_DIR / filename
        model = LoadedModel(key, MODEL_LABELS[key], path, None, False)
        try:
            if not path.exists():
                raise FileNotFoundError(f"No existe {filename}")
            if _is_lfs_pointer(path):
                raise RuntimeError(
                    f"{filename} es un pointer de Git LFS; ejecuta `git lfs pull` "
                    "para descargar el modelo entrenado."
                )
            with path.open("rb") as fh:
                model.estimator = pickle.load(fh)
            model.loaded = True
        except Exception as exc:  # noqa: BLE001 - se reporta al endpoint /health
            model.error = str(exc)
        loaded[key] = model
    return loaded


def startup() -> None:
    global _models, _metrics
    _metrics = load_metrics()
    _models = load_models()


def normalize_place(value: str) -> str:
    aliases = {
        "Álvaro Obregón": "AlvaroObregon",
        "Alvaro Obregon": "AlvaroObregon",
        "Benito Juárez": "BenitoJuarez",
        "Benito Juarez": "BenitoJuarez",
        "Miguel Hidalgo": "MiguelHidalgo",
        "Cuauhtémoc": "Cuauhtemoc",
        "Gustavo A. Madero": "GustavoAMadero",
        "Magdalena Contreras": "MagdalenaContreras",
        "La Magdalena Contreras": "MagdalenaContreras",
        "Venustiano Carranza": "VenustianoCarranza",
        "Milpa Alta": "MilpaAlta",
    }
    compact = str(value or "").strip()
    return aliases.get(compact, compact.replace(" ", ""))


def validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    property_type = str(payload.get("property_type", "")).strip()
    place_name = normalize_place(str(payload.get("place_name", "")).strip())
    algorithm = str(payload.get("algorithm", "lgbm")).strip().lower()

    if property_type not in PROPERTY_TYPES:
        errors.append("Selecciona un tipo de inmueble válido.")
    if place_name not in ZONE_AVERAGE_M2:
        errors.append("Selecciona una alcaldía válida de Ciudad de México.")
    if algorithm not in MODEL_FILES:
        errors.append("Selecciona un algoritmo válido: lgbm, rf o xgb.")

    def number(name: str, label: str, min_value: float, max_value: float) -> float:
        raw = payload.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} debe ser numérico.")
            return math.nan
        if not min_value <= value <= max_value:
            errors.append(f"{label} debe estar entre {min_value:g} y {max_value:g}.")
        return value

    surface_total = number("surface_total", "Superficie total", 10, 5000)
    surface_covered = number("surface_covered", "Superficie cubierta", 10, 5000)
    lat = number("lat", "Latitud", 19.0, 19.7)
    lon = number("lon", "Longitud", -99.4, -98.8)

    if not math.isnan(surface_total) and not math.isnan(surface_covered):
        if surface_covered > surface_total:
            errors.append("La superficie cubierta no puede ser mayor que la superficie total.")

    cleaned = {
        "property_type": property_type,
        "place_name": place_name,
        "surface_total": surface_total,
        "surface_covered": surface_covered,
        "lat": lat,
        "lon": lon,
        "algorithm": algorithm,
    }
    return cleaned, errors


def feature_frame(data: dict[str, Any], estimator: Any | None = None):
    """Construye el DataFrame con aliases para ajustarse al pipeline entrenado."""
    import pandas as pd

    aliases = {
        "property_type": data["property_type"],
        "place_name": data["place_name"],
        "surface_total": data["surface_total"],
        "surface_covered": data["surface_covered"],
        "surface_total_in_m2": data["surface_total"],
        "surface_covered_in_m2": data["surface_covered"],
        "lat": data["lat"],
        "lon": data["lon"],
        "latitude": data["lat"],
        "longitude": data["lon"],
    }

    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is None and hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            feature_names = getattr(step, "feature_names_in_", None)
            if feature_names is not None:
                break

    if feature_names is not None:
        row = {name: aliases.get(name, data.get(name, 0)) for name in feature_names}
        return pd.DataFrame([row], columns=list(feature_names))

    return pd.DataFrame(
        [
            {
                "property_type": data["property_type"],
                "place_name": data["place_name"],
                "surface_total_in_m2": data["surface_total"],
                "surface_covered_in_m2": data["surface_covered"],
                "lat": data["lat"],
                "lon": data["lon"],
            }
        ]
    )


def fallback_prediction(data: dict[str, Any]) -> float:
    """Estimación de respaldo para desarrollo cuando Git LFS no descargó modelos."""
    base_m2 = ZONE_AVERAGE_M2[data["place_name"]]
    type_factor = {"apartment": 1.0, "house": 0.92, "store": 1.15, "PH": 1.28}[data["property_type"]]
    covered_ratio = data["surface_covered"] / max(data["surface_total"], 1)
    return float(data["surface_total"] * base_m2 * type_factor * (0.9 + covered_ratio * 0.1))


def predict_with_model(key: str, data: dict[str, Any]) -> dict[str, Any]:
    model = _models[key]
    source = "trained_model" if model.loaded else "fallback"
    if model.loaded and model.estimator is not None:
        frame = feature_frame(data, model.estimator)
        raw_prediction = model.estimator.predict(frame)
        prediction = float(list(raw_prediction)[0])
    else:
        prediction = fallback_prediction(data)

    return {
        "algorithm": key,
        "label": model.label,
        "prediction": max(0, round(prediction, 2)),
        "formatted_price": format_currency(prediction),
        "metrics": _metrics[key],
        "model_loaded": model.loaded,
        "source": source,
        "warning": model.error if not model.loaded else None,
    }


def format_currency(value: float) -> str:
    return f"${round(value):,.0f} MXN"


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "success": True,
            "models": {
                key: {
                    "label": model.label,
                    "loaded": model.loaded,
                    "path": model.path.name,
                    "error": model.error,
                }
                for key, model in _models.items()
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data, errors = validate_payload(payload)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    selected_key = data["algorithm"]
    selected = predict_with_model(selected_key, data)
    all_predictions = {key: predict_with_model(key, data) for key in MODEL_FILES}
    zone_average = ZONE_AVERAGE_M2[data["place_name"]] * data["surface_total"]

    return jsonify(
        {
            "success": True,
            "input": data,
            "prediction": selected["prediction"],
            "formatted_price": selected["formatted_price"],
            "metrics": selected["metrics"],
            "selected_model": selected,
            "all_predictions": all_predictions,
            "zone_average": round(zone_average, 2),
            "formatted_zone_average": format_currency(zone_average),
        }
    )


startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
