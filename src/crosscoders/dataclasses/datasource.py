

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING




@dataclass
class DatasourceConfig:

    which: str = MISSING
    _target_: str = MISSING


@dataclass
class HuggingFaceDatasourceConfig(DatasourceConfig):

    _target_: str = 'crosscoders.data.dataset.HuggingFaceDatasource._load'

    org: str = '${dataset.org}'
    repo: str = '${dataset.repo}'
    slice: str = '${dataset.slice}'


@dataclass
class S3DatasourceConfig(DatasourceConfig):

    _target_: str = 'crosscoders.data.dataset.S3Datasource._load'

    bucket_name: str = '${globals.s3_bucket}'
    key_prefix: str = '/'.join('${paths.activations_dir}'.split('/')[3:])

