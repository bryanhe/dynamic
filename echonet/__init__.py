"""
The echonet package contains code for loading echocardiogram videos, and
functions for training and testing segmentation and ejection fraction
prediction models.
"""

from echonet.__version__ import __version__
from echonet.config import CONFIG as config
import echonet.datasets as datasets
import echonet.models as models
import echonet.utils as utils

import click


@click.group()
def main():
    pass


del click


main.add_command(utils.segmentation.run)
main.add_command(utils.video.run)

__all__ = ["__version__", "config", "datasets", "main", "models", "utils"]
