

import hydra

from omegaconf import OmegaConf

from crosscoders.data.dataset import Dataset




def main(cfg):

    # print('> loading dataset:')
    # print(OmegaConf.to_yaml(cfg.runner.dataset, resolve=True), end='\n\n')
    # ds = hydra.utils.instantiate(cfg.runner.dataset)

    ds = Dataset.instantiate(cfg.runner.dataset)

    print(ds)



    train_ds = ds.load()
    # print(train_ds.take_batch(5))

    ds.save(train_ds)
