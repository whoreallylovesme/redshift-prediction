import click

from lib.inference import inference


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
def main(config: str) -> None:
    inference(config_path=config)


if __name__ == "__main__":
    main()
