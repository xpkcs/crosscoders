



from crosscoders.data.dataset import Dataset
from crosscoders.runners.data import RayDataRunner


class TokensToActivationsDataRunner(RayDataRunner):

    def run(self) -> None:

        # self.cfg['dataset'][k] =
        ds = Dataset(self.cfg['dataset'])

        # print(ds)

        train_ds = ds.load('tokens')

        print(train_ds)
        # # print(train_ds.take_batch(5))

        # ds.save(train_ds)
