# Cotizador inmobiliario CDMX

Aplicación Flask + HTML para consumir tres modelos entrenados (`LightGBM`, `Random Forest` y `XGBoost`) y estimar el precio de inmuebles en Ciudad de México.

## Ejecutar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Si los .pkl son pointers de Git LFS, descargar primero:
git lfs pull
python app.py
```

Abre `http://localhost:5000`.

## Endpoints

- `GET /`: interfaz web responsive.
- `GET /health`: estado de carga de los modelos.
- `POST /predict`: inferencia en tiempo real.

Ejemplo de payload:

```json
{
  "property_type": "apartment",
  "place_name": "BenitoJuarez",
  "surface_total": 100,
  "surface_covered": 100,
  "lat": 19.3682,
  "lon": -99.1717,
  "algorithm": "lgbm"
}
```

## Uso de LLM

Se utilizó un modelo de lenguaje como asistente de ingeniería para estructurar el backend Flask, conectar la interfaz HTML con el endpoint `/predict`, proponer validaciones de entrada, diseñar la visualización comparativa y documentar el flujo de ejecución. El criterio del equipo se mantuvo en la selección de rangos válidos, métricas reportadas y revisión del código final.
