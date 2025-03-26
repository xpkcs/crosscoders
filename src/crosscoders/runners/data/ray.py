



from crosscoders.data.dataset import Dataset
from crosscoders.runners.data import RayDataRunner

from crosscoders.config import Config, get_config

CONFIG: Config = get_config()




class TokensToActivationsDataRunner(RayDataRunner):

    def run(self) -> None:

        ds = Dataset(CONFIG.dataset)

        # print(ds)

        train_ds = ds.load('tokens')

        print(train_ds)
        # # print(train_ds.take_batch(5))

        ds.save(train_ds)
