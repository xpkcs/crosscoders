# crosscoders

A Python package to run crosscoders.


## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Run Configs](#run-configs)
3. [Usage](#usage)

<!--  -->

## Installation

1. Install conda

2. `conda env create --file=env.yml`

<!--  -->

## Configuration

#### Environment Variables
Create a `.env` file in the repo root directory and add the following lines, modifying `PROJECT_ROOT_DIR` and `CONFIG_FILEPATH` to point to the **absolute** path of the repo root directory and the run config file you want to use.

```sh
export PROJECT_ROOT_DIR="."
export CONFIG_FILEPATH="${PROJECT_ROOT_DIR}/src/scripts/configs/cfg.yml"   # eg
```

> _See `src/crosscoders/constants.py` for a list of `REQUIRED_ENV_VARS`._

> _See `src/scripts/configs/` for preset run configurations._

#### Run Configs
The run config files are the primary place to tweak anything in the code. The syntax is defined by the config dataclasses at `src/crosscoders/dataclasses/configs/`.

Currently, there are two base config objects that contain all others: `GlobalsConfig` and `RunnerConfig`. Your `cfg.yml` should have only two root dictionaries: `GLOBALS` and `RUNNER`.

For the most part, the field names in the `cfg.yml` are uppercase versions of the config dataclass names minus `Config`.

> _See `src/crosscoders/dataclasses/configs/`._

<!--  -->

## Usage
This package is intended to be used as a CLI called `xc`, although it can be used programmatically as well.

There is only one required argument, `mode`, which specifies which sub-script to run.
<!-- You can also specify a path to custom run configs. -->

> _See `src/scripts/main.py`._

```sh
# export CONFIG_FILEPATH="${PROJECT_ROOT_DIR}/src/scripts/configs/data.yml
xc data
```

```sh
# export CONFIG_FILEPATH="${PROJECT_ROOT_DIR}/src/scripts/configs/train.yml
xc train
```