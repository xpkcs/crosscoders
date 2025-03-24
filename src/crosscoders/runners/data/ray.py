



from crosscoders.runners.data import RayDataRunner


class TokensToActivationsDataRunner(RayDataRunner):

    def run(self) -> None:

        ds = self.cfg['dataset']

        train_ds = ds.load()

        print(train_ds)
        # print(train_ds.take_batch(5))

        ds.save(train_ds)
