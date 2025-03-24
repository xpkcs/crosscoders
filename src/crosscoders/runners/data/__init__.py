

from crosscoders.runners import Runner, RayRunner, SparkRunner




class DataRunner(Runner):
    ...


class RayDataRunner(DataRunner, RayRunner):
    ...


class SparkDataRunner(DataRunner, SparkRunner):
    ...
