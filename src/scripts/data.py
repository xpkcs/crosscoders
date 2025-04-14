

from pprint import pformat, pprint
from crosscoders.data.dataset import Dataset
from crosscoders.config import *
from crosscoders.utils import delete_files_in_s3
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import ray




def make_placement_group(bundles, strategy='STRICT_SPREAD'):


    pg = ray.util.placement_group(
        bundles=bundles,
        strategy=strategy
    )

    ray.get(pg.ready())
    print('placement group is ready')

    # add a small delay to ensure scheduler registration
    import time;        time.sleep(5)


    return pg




def main(cfg):

    CONFIG: Config = get_config()

    delete_files_in_s3(CONFIG.globals.s3_bucket, '/'.join(CONFIG.paths.activations_dir.split('/')[3:]))


    num_gpus = 4
    pg = make_placement_group([{'GPU': 1, 'CPU': 1}] * num_gpus)


    ds = Dataset(CONFIG.dataset)

    ray_ds, n_tokens = ds.load('tokens', PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=-1))


    ds.save(ray_ds)
