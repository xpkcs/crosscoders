

from crosscoders.data.dataset import TinyStoriesRayDataset
# from crosscoders import CONSTANTS


def main():

    # file_slice = (0, 100)
    # train_ds = TinyStoriesRayDataset(s3_prefix=f'tiny-stories-33M/chunks/{file_slice[0]}-{file_slice[1]}/')
    train_ds = TinyStoriesRayDataset(s3_prefix='tiny-stories-33M/100M/')
    train_ds.save(train_ds.load())





# if __name__ == '__main__':
#     main()
