

# from crosscoders.data.dataset import Dataset
# from crosscoders.utils import instantiate




from abc import abstractmethod

from crosscoders.runners import RayRunner, Runner


class DataRunner(Runner):
    ...

class RayDataRunner(DataRunner, RayRunner):
    ...


