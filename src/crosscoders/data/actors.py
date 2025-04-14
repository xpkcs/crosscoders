

import ray

from crosscoders.config import get_config
from crosscoders.io import ZarrIO




ceil = lambda n: int(n) if n == int(n) else int(n) + 1


CONFIG = get_config()






@ray.remote(num_cpus=2, memory=7 * 1024 * 1024 * 1024)
class ZarrActor:

    def __init__(self, **kwargs):

        self.root, self.zarrays = ZarrIO.init(CONFIG.paths.zarr_dir, **kwargs)


    def get_root(self):

        return self.root


    def get_array(self, layer, activation_type, start_idx, end_idx):

        return self.zarrays['activations'][(layer, activation_type)][start_idx:end_idx]




@ray.remote(max_concurrency=1)
class Indexer:

    def __init__(self):

        self.reset()

    def reset(self):

        self.n = 0

    def get(self, attr):

        return getattr(self, attr)

    def reserve(self, n: int) -> int:

        start_idx = self.n
        self.n += n

        return start_idx



@ray.remote(max_concurrency=1)
class ChunkIndexer:

    def __init__(self, chunk_size):

        self.chunk_size = chunk_size
        self.reset()

    def reset(self):

        self.n_chunks = 0

    def get(self, attr):

        return getattr(self, attr)

    def reserve(self, n_tokens_req: int) -> int:

        start_idx = self.n_chunks * self.chunk_size

        n_chunks_req = ceil(n_tokens_req / self.chunk_size)
        self.n_chunks += n_chunks_req

        return start_idx, start_idx + (n_chunks_req * self.chunk_size)
