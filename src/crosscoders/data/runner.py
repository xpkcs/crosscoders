

# from crosscoders.data.dataset import Dataset
# from crosscoders.utils import instantiate




from abc import abstractmethod


class DataRunner:

    def __init__(self, **kwargs: dict) -> None:

        self.cfg = kwargs

        # for k, v in kwargs.items():
        #     setattr(self, k, v)


    @abstractmethod
    def run(self) -> None:
        ...


class TokensToActivationsDataRunner(DataRunner):

    def run(self) -> None:

        ds = self.cfg['dataset']

        train_ds = ds.load()

        print(train_ds)
        # print(train_ds.take_batch(5))

        ds.save(train_ds)


class SparkDataRunner(DataRunner):
    ...
