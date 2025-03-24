

from crosscoders.utils import instantiate




def main(cfg):


    # cfg.dataset.batch_size = '${runner.batch_size}'
    # cfg.dataset.max_tokens = '${runner.max_tokens}'


    # copy dataset config under runner's so that runner can use it
    # cfg.runner.dataset = '${dataset}'

    runner = instantiate(cfg.runner, True)
    runner.run()




    # print('> loading dataset:')
    # print(OmegaConf.to_yaml(cfg.runner.dataset, resolve=True), end='\n\n')
    # ds = hydra.utils.instantiate(cfg.runner.dataset)

    # ds = Dataset.instantiate(cfg.runner.dataset)

    # train_ds = ds.load()

    # print(train_ds)
    # # print(train_ds.take_batch(5))

    # ds.save(train_ds)



