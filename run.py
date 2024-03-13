import click

from pydentification.experiment.entrypoints import run  # isort:skip
from pydentification.experiment.entrypoints import sweep as sweep_entrypoint  # isort:skip

import src as runtime  # alias entire module with required functions  # isort:skip


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
