# Architecture / Design

## High-level modules
- `core/`: domain models (`RasterValidation`, `VectorValidation`, `BenchmarkResult`) and exceptions.
- `services/`: pure workflow logic (validation, audit, benchmark, diagnostics).
- `api/`: CLI adapters and command router (`python -m api.cli`).
- `utils/`: bounded helper utilities (filesystem-safe access).
- `config/`: environment-driven defaults.
- `task1-path-tracing/` and `task2-viewsheds/`: primary domain pipelines.

## Design principles
1. **Separation of concerns**: command parsing stays in `api/`, computation in `services/`.
2. **Backward compatibility**: root scripts remain executable and delegate to the modular layer.
3. **Pragmatic reliability**: lazy imports for heavy geospatial deps reduce import-time failures.
4. **Traceable validation**: acceptance checks and outputs are kept as first-class artifacts.

## Runtime flow
1. User executes script or `api.cli` command.
2. API layer validates arguments and delegates to service.
3. Service performs domain computation and writes report/output artifacts.
4. Acceptance check scripts verify required and optional deliverables.

## Tradeoffs
- Security audit is heuristic-based, not full static taint analysis.
- ML diagnostics remain lightweight and tied to available training outputs.
- Existing task outputs are reused for submission verification instead of full regeneration in this package run.
