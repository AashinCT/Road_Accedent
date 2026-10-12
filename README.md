# AI-Based Road Accident Prediction

A student micro-project MVP for exploring road-accident records, mapping geolocated incidents, and evaluating a machine-learning model when a suitable dataset is supplied.

## Current scope
- Interactive Leaflet map with OpenStreetMap tiles.
- FastAPI backend for dataset inspection, summary metrics, incident locations, and model training/prediction.
- CSV-based workflow so the project can use a verified Indian accident dataset.
- Clear distinction between severity classification and accident-occurrence risk.

## Important limitations
This is a prototype, not a road-safety or emergency system. A severity model estimates severity conditional on a recorded accident; it does not by itself estimate the probability that an accident will happen at a selected map point. Point-specific risk requires trustworthy location coordinates and suitable non-accident/exposure data. Do not use the prototype to make real-world safety decisions.

## Requirements
- Python 3.10+
- pip

## Run locally
```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000 in your browser.

## Add a dataset
Place a CSV at `data/accidents.csv`. The API supports common latitude/longitude column names and will show records on the map when valid coordinates are present. Use `/api/dataset/inspect` to inspect the dataset. For model training, select a target column that actually represents accident severity or another documented label. The API will reject missing or unsuitable targets rather than invent predictions.

## API
- `GET /api/health`
- `GET /api/summary`
- `GET /api/incidents`
- `GET /api/dataset/inspect`
- `POST /api/model/train` with JSON `{"target_column":"..."}`
- `POST /api/model/predict` with JSON `{"features":{"column":"value"}}`

## Dataset note
The dataset is not bundled. Confirm the source, geography, column definitions, license, and presence of coordinates before reporting results. Never describe demo/sample records as real accident reports.
