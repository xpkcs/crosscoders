

# from crosscoders.data.dataset import Dataset
# from crosscoders.utils import instantiate




class DataRunner:

    def __init__(self, **kwargs):
        # for k, v in kwargs.items():
        #     setattr(self, k, v)
        self.cfg = kwargs


class TokensToActivationsDataRunner(DataRunner):

    def run(self):

        ds = self.cfg['dataset']

        train_ds = ds.load()

        print(train_ds)
        # print(train_ds.take_batch(5))

        # ds.save(train_ds)


class SparkDataRunner(DataRunner):

    def __init__(self):
        super().__init__()