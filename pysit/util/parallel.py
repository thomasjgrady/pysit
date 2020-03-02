try:
    from mpi4py import MPI
    hasmpi = True
except:
    hasmpi = False

__all__ = ['hasmpi', 'ParallelWrapShotNull', 'ParallelWrapShot']

# Mapping between dimension and the key labels for Cartesian domain
_cart_keys = {1: [(0, 'z')],
              2: [(0, 'x'), (1, 'z')],
              3: [(0, 'x'), (1, 'y'), (2, 'z')]}

class ParallelWrapShotBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('ParallelWrapShotBase.__init__ should never be called.')

class ParallelWrapShotNull(ParallelWrapShotBase):

    def __init__(self, *args, **kwargs):
        self.comm = None
        self.use_parallel = False

        self.size = 1
        self.rank = 0

class ParallelWrapShot(ParallelWrapShotBase):

    def __new__(cls, *args, **kwargs):

        if not hasmpi:
            return ParallelWrapShotNull(*args, **kwargs)

        if MPI.COMM_WORLD.Get_size() <= 1:
            return ParallelWrapShotNull(*args, **kwargs)

        return super().__new__(cls)

    def __init__(self, comm=None, *args, **kwargs):
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        self.use_parallel = True
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

class ParallelWrapCartesianBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('ParallelWrapShotBase.__init__ should never be called.')

class ParallelWrapCartesianNull(ParallelWrapCartesianBase):
    
    def __init__(self, *args, **kwargs):
        self.comm = None
        self.use_parallel = False

        self.size = 1
        self.rank = 0

class ParallelWrapCartesian(ParallelWrapCartesianBase):
    """
    Class for wrapping an MPI communicator for cartesian topologies
    """

    def __new__(cls, *args, **kwargs):

        if not hasmpi:
            return ParallelWrapCartesianNull(*args, **kwargs)
            
        if MPI.COMM_WORLD.Get_size() <= 1:
            return ParallelWrapCartesianNull(*args, **kwargs)
        
        return super().__new__(cls)

    def __init__(self, dims, periods=None, reorder=False, comm=None, *args, **kwargs):
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        self.use_parallel = True
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        
        # Get the dims used for decomposition
        self.dim = len(dims)
        self.dims = dims

        # If no periods are passed, assume non periodic by default
        if periods is None:
            self.periods = [False] * self.dim
        else:
            self.periods = periods
        
        # Set reorder
        self.reorder = reorder

        # Create a new cartesian communicator
        self.cart_comm = self.comm.Create_cart(dims=self.dims, periods=self.periods,
                reorder=self.reorder)
        
        # Get the size and rank from communicator
        self.cart_size = self.cart_comm.Get_size()
        self.cart_rank = self.cart_comm.Get_rank()

        # Get coordinates of this rank
        self.cart_coords = self.cart_comm.Get_coords(self.cart_rank)
        
        # Get the coordinates and ranks of neighbors
        self.neighbor_coords = [None] * self.dim
        self.neighbor_ranks  = [None] * self.dim
        for (i, _) in _cart_keys[self.dim]:
            lcoords = [x-1 if j == i else x for j, x in enumerate(self.cart_coords)]
            rcoords = [x+1 if j == i else x for j, x in enumerate(self.cart_coords)]

            self.neighbor_coords[i] = [lcoords, rcoords]

            if -1 in lcoords:
                lrank = MPI.PROC_NULL
            else:
                lrank = self.cart_comm.Get_cart_rank(lcoords)

            if self.dims[i] in rcoords:
                rrank = MPI.PROC_NULL
            else:
                rrank = self.cart_comm.Get_cart_rank(rcoords)

            self.neighbor_ranks[i] = [lrank, rrank]
    
    def split_discrete(self, vals):
        """
        Splits a list of integers vals along each of the axes

        Parameters
        ----------

        vals : list of integers
            List of values to split. Must be the same dimension as the cartesian
            topology

        Returns
        -------

        ns : List of integers
            List of split values

        os: List of integers
            List of offsets of split values
        """

        ns = list()
        os = list()

        for c, d, v in zip(self.cart_coords, self.dims, vals):
            r = v % d
            if c < r:
                n = v // d + 1
                o = n * c
            else:
                n = v // d
                o = r * (n + 1) + (c - r) * n
            ns.append(n)
            os.append(o)

        return ns, os


if __name__ == '__main__':

#   x = ParallelWrapShotBase()
    y = ParallelWrapShotNull()
    z = ParallelWrapShot()

    c1 = ParallelWrapCartesianNull()
    c2 = ParallelWrapCartesian(dims=(2,2))
