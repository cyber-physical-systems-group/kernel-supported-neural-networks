import logging
from functools import partial
from typing import Any

import click
import wandb
import yaml

from src.runner.context import RuntimeContext
from src.runner.parameters import left_dict_join, prepare_config_for_sweep

import src.runner as runtime  # alias entire module with required functions  # isort:skip


def run_training(
    runtime: RuntimeContext,
    project_name: str,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    checkpoint_path: str | None = None,
):
    """
    This function is used to run a single training experiment with given configuration. It contains the main
    experimentation logic and parameter passing.
    """
    for config in (dataset_config, model_config, training_config):
        if isinstance(config, dict):  # prevents logging parameters twice in sweep mode
            wandb.log(config)

    try:
        # merge static and dynamic parameters of dataset
        data_parameters = left_dict_join(training_config, model_config)
        # experiment flow
        dm = runtime.input_fn(dataset_config, data_parameters)
        model, trainer = runtime.model_fn(project_name, training_config, model_config, checkpoint_path)
        model, trainer = runtime.train_fn(model, trainer, dm, checkpoint_path)
        runtime.report_fn(model, trainer, dm)
        runtime.save_fn(wandb.run.id, model)

    except Exception as e:
        logging.exception(e)  # log traceback, W&B can sometimes lose information
        raise ValueError("Experiment failed.") from e


def run_sweep_step(
    runtime: RuntimeContext, project_name: str, dataset_config: dict[str, Any], experiment_config: dict[str, Any]
):
    with wandb.init(reinit=True):
        wandb.mark_preempting()
        parameters = wandb.config  # dynamically generated model settings by W&B sweep

        run_training(
            runtime=runtime,
            project_name=project_name,
            dataset_config=dataset_config,
            model_config=parameters,
            training_config=experiment_config["training"],
        )


def run(data: str, experiment: str, resume: str, runtime: RuntimeContext):
    """
    Run single experiment with given configuration.

    :param data: dataset configuration
    :param experiment: experiment configuration
    :param runtime: runtime context with code executing the training and all preparations
    :param resume: resume training from a given run_id, this needs to load existing model from checkpoint and set
                   the training state and optimizer state correctly, otherwise unexpected behavior may occur
    """
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    if resume:
        resume_config = yaml.safe_load(open(resume))
    else:
        resume_config = {}

    project = experiment_config["general"]["project"]
    name = experiment_config["general"]["name"]
    resume = resume_config.get("resume_mode")  # will be None if resume not used
    run_id = resume_config.get("run_id")

    with wandb.init(project=project, name=name, resume=resume, id=run_id):
        model_config = experiment_config["model"]
        training_config = experiment_config["training"]

        run_training(
            runtime=runtime,
            project_name=project,
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            checkpoint_path=resume_config.get("checkpoint_path"),
        )


def sweep(data: str, experiment: str, runtime: RuntimeContext):
    """
    Run a sweep experiment with given configuration.

    :param data: dataset configuration
    :param experiment: experiment configuration
    :param runtime: runtime context with code executing the training and all preparations
    """
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    sweep_parameters = left_dict_join(experiment_config["sweep_parameters"], experiment_config["model"])
    sweep_config = prepare_config_for_sweep(experiment_config["sweep"], sweep_parameters)
    sweep_id = wandb.sweep(sweep_config, project=experiment_config["general"]["project"])

    run_sweep_fn = partial(
        run_sweep_step, runtime, experiment_config["general"]["project"], dataset_config, experiment_config
    )
    wandb.agent(
        sweep_id,
        function=run_sweep_fn,
        count=experiment_config["general"]["n_runs"],
        project=experiment_config["general"]["project"],
    )


@click.command()
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--experiment", type=click.Path(exists=True), required=True)
@click.option("--sweep", type=bool, is_flag=True, default=False)
def main(data: str, experiment: str, sweep: bool):
    if sweep:
        sweep_entrypoint(data=data, experiment=experiment, runtime=runtime)  # noqa
    else:
        run(data=data, experiment=experiment, runtime=runtime)  # noqa


if __name__ == "__main__":
    main()
