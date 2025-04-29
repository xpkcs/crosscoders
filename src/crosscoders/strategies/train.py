


from crosscoders.data.dataset import Dataset


class DefaultTrainStrategy:

    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self, fit_loop):

        ds = Dataset(self.cfg.dataset, 'activations')
        print(ds.schema())

        # train_dl = None
        # val_dl = None
        # model = None
        # loss_fn = None
        # optimizer = None
        # schedulers = None

        # # return fit_loop(train_dl, val_dl, model, loss_fn, optimizer, schedulers, cfg)
        # return fit_loop(model)
