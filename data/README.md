# Local accident dataset

Put your verified accident CSV here as `accidents.csv` (this file is intentionally not committed).

Before using it, check the source, license, geographic coverage, column definitions, and missing values. The API recognizes common coordinate columns such as `latitude`/`longitude` or `lat`/`lon`. If the dataset has no coordinates, it can still be inspected and used for a suitable classification task, but the map cannot display its records.

Do not label an accident-severity classifier as a model that predicts whether an accident will happen. Point-specific occurrence risk needs appropriate non-accident/exposure examples and validated geospatial/time features.
