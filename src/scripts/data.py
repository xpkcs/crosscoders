

from crosscoders.data.dataset import TinyStoriesRayDataset
# from crosscoders import CONSTANTS


def main():

    train_ds = TinyStoriesRayDataset()
    train_ds.save(train_ds.load())





# if __name__ == '__main__':
#     main()
