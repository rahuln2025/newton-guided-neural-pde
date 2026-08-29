"""
MLflow helpers for experiment tracking.

Designed to be non-invasive: all functions are no-ops when
mlflow_config.enabled is False or mlflow is not installed.

Usage in training loops:
    mlflow_run = start_mlflow_run(config)
    ...
    log_mlflow_metrics(mlflow_run, {'train_loss': ..., 'val_loss': ...}, step=epoch)
    log_mlflow_artifact(mlflow_run, path)
    end_mlflow_run(mlflow_run)
"""

import os
import yaml


def start_mlflow_run(config):
    """
    Start an MLflow run if mlflow_config.enabled is True.

    Returns an mlflow.ActiveRun context or None.
    """
    mlflow_cfg = config.get('mlflow_config', {})
    if not mlflow_cfg.get('enabled', False):
        return None

    try:
        import mlflow
    except ImportError:
        print("WARNING: mlflow not installed. Skipping MLflow tracking.")
        return None

    tracking_uri = mlflow_cfg.get('tracking_uri', 'sqlite:///mlruns/mlflow.db')

    # Ensure parent directory exists for SQLite URIs
    if tracking_uri.startswith('sqlite:///') and not tracking_uri.startswith('sqlite:////'):  
        db_path = tracking_uri[len('sqlite:///'):]  # relative path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

    experiment_name = mlflow_cfg.get('experiment_name', 'default')
    run_name = mlflow_cfg.get('run_name') or config.get('training_config', {}).get('log_name', None)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=run_name)
    print(f"MLflow run started: {run.info.run_id} | experiment: {experiment_name} | tracking: {tracking_uri}")

    # Log flat params from config
    training_config = config.get('training_config', {})
    data_config = config.get('data_config', {})
    model_config = config.get('model_config', {})
    solver_config = config.get('solver_config', {})

    params = {
        # ── Run identity ──────────────────────────────────────────────────
        'run_id':           config.get('run_id'),
        'run_name':         config.get('run_name'),
        # ── Training ─────────────────────────────────────────────────────
        'lr':               training_config.get('learning_rate'),
        'batch_size':       training_config.get('batch_size'),
        'epochs':           training_config.get('epochs'),
        'precision':        training_config.get('precision'),
        'weight_decay':     training_config.get('weight_decay'),
        'use_lr_scheduler': training_config.get('use_lr_scheduler'),
        'training_type':    data_config.get('training_type'),
        # ── Data ─────────────────────────────────────────────────────────
        'n_pts':            data_config.get('n_pts'),
        'total_samples':    data_config.get('total_num_samples'),
        'train_frac':       data_config.get('train_frac'),
        'val_frac':         data_config.get('val_frac'),
        # ── Model ────────────────────────────────────────────────────────
        'model_type':       model_config.get('model_type'),
        'in_channels':      model_config.get('in_channels'),
        'out_channels':     model_config.get('out_channels'),
        'hidden_channels':  model_config.get('hidden_channels'),
        'n_layers':         model_config.get('n_layers'),
        'n_modes':          model_config.get('n_modes'),
        'norm':             model_config.get('norm'),
        'activation':       model_config.get('activation'),
    }

    # ── Solver config ─────────────────────────────────────────────────────
    if solver_config:
        params.update({
            'maxiters':           solver_config.get('maxiters'),
            'max_maxiters':       solver_config.get('max_maxiters'),
            'min_maxiters':       solver_config.get('min_maxiters'),
            'tol':                solver_config.get('tol'),
            'solver_type':        solver_config.get('solver_type'),
            'n_mesh':             solver_config.get('n_mesh'),
            'curriculum_enabled': solver_config.get('curriculum_enabled'),
            'alpha_init':         solver_config.get('alpha_init'),
            'alpha_max':          solver_config.get('alpha_max'),
        })

    # ── Scheduler ─────────────────────────────────────────────────────────
    sched_cfg = training_config.get('scheduler_config', {})
    if sched_cfg:
        params['scheduler_step']  = sched_cfg.get('step_size')
        params['scheduler_gamma'] = sched_cfg.get('gamma')

    # Filter None values and log
    params = {k: v for k, v in params.items() if v is not None}
    mlflow.log_params(params)

    # Log full config YAML as a tag (no 250-char limit, survives UI inspection)
    mlflow.set_tag('config_yaml', yaml.dump(config, default_flow_style=False, sort_keys=False))

    return run


def log_mlflow_metrics(mlflow_run, metrics: dict, step: int):
    """Log a dict of scalar metrics at the given step."""
    if mlflow_run is None:
        return
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        print(f"MLflow metric logging failed: {e}")


def log_mlflow_artifact(mlflow_run, path: str):
    """
    Log a file or directory as an MLflow artifact.
    Silently skips if path does not exist.
    """
    if mlflow_run is None or not os.path.exists(path):
        return
    try:
        import mlflow
        if os.path.isdir(path):
            mlflow.log_artifacts(path, artifact_path=os.path.basename(path))
        else:
            mlflow.log_artifact(path)
    except Exception as e:
        print(f"MLflow artifact logging failed ({path}): {e}")


def log_mlflow_config(mlflow_run, config: dict, checkpoint_dir: str):
    """Log the config yaml file from the checkpoint directory."""
    if mlflow_run is None:
        return
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    log_mlflow_artifact(mlflow_run, config_path)


def end_mlflow_run(mlflow_run):
    """End the active MLflow run."""
    if mlflow_run is None:
        return
    try:
        import mlflow
        mlflow.end_run()
        print("MLflow run ended.")
    except Exception as e:
        print(f"MLflow end_run failed: {e}")
