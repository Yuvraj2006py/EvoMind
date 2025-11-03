# EvoMind Configuration Reference


## Experiment

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | `EvoMind` | Display name used in logs, reports, and dashboards. |
| `mlflow_uri` | `Optional[str]` | `None` | MLflow tracking URI (e.g. file:./mlruns) used for experiment logging. |
| `mlflow_tracking_uri` | `Optional[str]` | `None` | Legacy alias for mlflow_uri. |

## Engine

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `generations` | `int` | `5` | Number of evolutionary generations to execute. |
| `population` | `int` | `20` | Number of genomes retained per generation. |
| `epochs` | `int` | `5` | Training epochs per genome during evolution. |
| `batch_size` | `int` | `32` | Mini-batch size for the Torch trainer. |
| `learning_rate` | `float` | `0.001` | Optimiser learning rate used during genome training. |
| `parallel` | `bool` | `True` | Enable Ray-based distributed evaluation when available. |
| `elite_fraction` | `float` | `0.2` | Fraction of top-performing genomes copied into the next generation. |
| `mutation_rate` | `float` | `0.3` | Probability of applying mutations to offspring genomes. |
| `crossover_rate` | `float` | `0.6` | Probability that offspring are produced via crossover. |
| `bayes_rounds` | `int` | `0` | Optuna tuning trials executed on elite genomes each generation. |
| `ensemble_top_k` | `int` | `3` | Number of top genomes combined into the final ensemble prediction. |
| `max_workers` | `Optional[int]` | `None` | Max worker threads used when Ray is unavailable (None = auto). |

## Trainer

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `epochs` | `int` | `5` | Baseline epochs used when engine.epochs is not provided. |
| `batch_size` | `int` | `32` | Baseline batch size used when engine.batch_size is not provided. |
| `learning_rate` | `float` | `0.001` | Baseline learning rate used when engine.learning_rate is not provided. |

## Data

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `sensitive_feature` | `Optional[str]` | `None` | Feature used to compute fairness diagnostics (e.g. demographic parity). |

## Reporting

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `export_formats` | `List[str]` | `['html', 'pdf']` | Report formats generated after a run. |

## Insights

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `shap_sample_size` | `int` | `200` | Number of rows sampled when generating SHAP explanations. |

## Fitness

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `accuracy_weight` | `float` | `1.0` | Weight applied to accuracy-style metrics when computing fitness. |
| `loss_weight` | `float` | `0.4` | Weight applied to validation loss when computing fitness. |
| `latency_weight` | `float` | `0.1` | Weight applied to latency penalty when computing fitness. |
| `robustness_weight` | `float` | `0.3` | Weight applied to robustness metrics when computing fitness. |

## Scheduler

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `generations` | `int` | `5` | Legacy scheduler configuration mapping to engine.generations. |
| `run_name` | `str` | `retail-forecasting-demo` | Optional run name used in legacy configs. |
| `log_history` | `bool` | `True` | Whether to log per-generation history (legacy compatibility). |

## Evolution

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `elite_fraction` | `float` | `0.2` | Legacy alias for engine.elite_fraction. |
| `mutation_rate` | `float` | `0.3` | Legacy alias for engine.mutation_rate. |
| `crossover_rate` | `float` | `0.6` | Legacy alias for engine.crossover_rate. |

## Legacy

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `population_size` | `int` | `20` | Backwards-compatible alias for engine.population. |
| `scheduler_generations` | `int` | `5` | Backwards-compatible alias for engine.generations. |
| `evolution_elite_fraction` | `float` | `0.2` | Backwards-compatible alias for engine.elite_fraction. |
| `evolution_mutation_rate` | `float` | `0.3` | Backwards-compatible alias for engine.mutation_rate. |
| `evolution_crossover_rate` | `float` | `0.6` | Backwards-compatible alias for engine.crossover_rate. |
