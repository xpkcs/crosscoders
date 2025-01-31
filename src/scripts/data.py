







from crosscoders.data.dataset import TinyStoriesRayDataset
# from crosscoders import CONSTANTS


def main():

    ds = TinyStoriesRayDataset()
    ds.save(ds.load())





if __name__ == '__main__':
    main()
