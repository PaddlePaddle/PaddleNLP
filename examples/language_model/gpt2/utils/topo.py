import numpy as np
from collections import namedtuple

GroupInfo = namedtuple('GroupInfo', ['size', 'rank', 'world'])


class Topology:
    def __init__(self, rank, world_size, dp, pp, sharding, mp):
        arr = np.arange(0, dp * pp * sharding * mp).reshape(
            [dp, pp, sharding, mp])

        idp, ipp, isharding, imp = np.where(arr == rank)
        idp = idp[0]
        ipp = ipp[0]
        isharding = isharding[0]
        imp = imp[0]

        self.world = GroupInfo(
            size=world_size, rank=rank, world=list(range(0, world_size)))

        mp = arr[idp, ipp, isharding, :]
        self.mp = GroupInfo(size=len(mp), rank=imp, world=mp.tolist())

        sharding = arr[idp, ipp, :, imp]
        self.sharding = GroupInfo(
            size=len(sharding), rank=isharding, world=sharding.tolist())

        pp = arr[idp, :, isharding, imp]
        self.pp = GroupInfo(size=len(pp), rank=ipp, world=pp.tolist())

        dp = arr[:, ipp, isharding, imp]
        self.dp = GroupInfo(size=len(dp), rank=idp, world=dp.tolist())

        self.is_last = self.pp.rank == self.pp.size - 1

        self.data_worldsize = self.dp.size * self.sharding.size
        self.data_inner_times = self.world.size // self.data_worldsize
        self.data_rank = self.dp.rank * self.sharding.size + self.sharding.rank

    def __repr__(self):
        return f'dp:\n\t {self.dp}, \npp:\n\t {self.pp}, \nsharding:\n\t {self.sharding}, \nmp:\n\t {self.mp}'
