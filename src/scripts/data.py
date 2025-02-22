

from crosscoders.data.dataset import TinyStoriesRayDataset
# from crosscoders import CONSTANTS


def main():

    train_ds = TinyStoriesRayDataset(s3_prefix='tiny-stories-33M/mlp_out/')
    train_ds.save(train_ds.load())





# if __name__ == '__main__':
#     main()
