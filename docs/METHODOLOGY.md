# Methodology

## Research Objective

RaceCast investigates whether publicly accessible Formula 1 telemetry can be used to decompose segment-level lap-time differences and evaluate how analytical and machine-learning methods transfer across circuits.

The study focuses on the question:

> Where is lap time gained or lost, what telemetry characteristics are associated with those differences, and how well do those relationships transfer to an unseen circuit?

## Data Source

Telemetry was obtained through FastF1 from 2024 Formula 1 qualifying sessions at:

- Bahrain
- Monaco
- Monza
- Silverstone
- Suzuka

The final dataset contains:

- 372 same-driver lap comparisons
- 2,487 segment-level observations
- 5 circuits
- 21 drivers represented

## Lap Selection

Valid qualifying laps with usable telemetry were selected.

For each driver and circuit, the fastest selected lap was used as the reference lap. Other selected laps from the same driver were compared against that reference.

## Distance Alignment

Telemetry was aligned by circuit distance rather than timestamp.

The analysis uses a 5 m spatial grid.

This allows telemetry from two laps to be compared at approximately the same physical locations around the circuit.

## Segment Construction

Reference-lap speed traces were smoothed and local speed minima were used to identify corner-centered segments.

Segment boundaries were derived from the relative positions of detected minima.

This is an automated telemetry-based segmentation method and should not be interpreted as an official circuit corner definition.

## Features

### Speed-related
- Entry-speed delta
- Minimum-speed delta
- Mean-speed delta
- Exit-speed delta
- Exit-acceleration proxy

### Braking
- Brake fraction delta
- Relative brake-start delta
- Braking-zone-length delta

### Throttle
- Mean throttle delta
- Full-throttle fraction delta
- Throttle-reapplication delta

### Powertrain and vehicle state
- RPM delta
- Gear delta
- Gear-change delta
- DRS fraction delta

### Segment context
- Segment length
- Reference segment time

## Target

The target variable is segment traversal-time delta:

comparison segment time minus reference segment time.

Positive values represent time lost relative to the reference segment.

## Models

The evaluated learned models were:

- Linear Regression
- Random Forest Regression
- Histogram Gradient Boosting

## Validation

Leave-One-Circuit-Out validation was used.

For each fold:

- Four circuits were used for training.
- One circuit was held out entirely for testing.

This was repeated until every circuit had served as the unseen test circuit.

## Evaluation Metrics

The study reports:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Spearman correlation
- Feature-group ablation
- Permutation importance
- SHAP analysis

## Analytical Baseline

A non-learned analytical reconstruction was also evaluated using the relationship between segment distance, reference traversal time, and mean-speed difference.

This analytical method achieved lower error than the evaluated machine-learning models.

## Methodological Limitations

The study uses public telemetry rather than proprietary team telemetry.

Important limitations include:

- Speed-derived variables are mathematically related to segment traversal time.
- Public brake information is less detailed than professional telemetry.
- Automated segment boundaries are not canonical circuit definitions.
- Observational telemetry does not establish causality.
- Vehicle setup, exact fuel load, tire state, track evolution and other latent variables cannot be fully controlled.

The results should therefore be interpreted as telemetry decomposition and comparative modeling rather than causal vehicle-performance analysis.
