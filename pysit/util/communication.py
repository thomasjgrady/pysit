try:
    from mpi4py import MPI
    hasmpi = True
except ImportError:
    hasmpi = False

import numpy as np

_cart_keys = {1: [(0, 'z')],
              2: [(0, 'x'), (1, 'z')],
              3: [(0, 'x'), (1, 'y'), (2, 'z')]}

def create_slices(pads):
    """
    Creates the slice objects used to index an array during ghost exchange
    pattern using the pads used to pad an array.

    Parameters
    ----------

    pads : list or list-like
        A list of tuples of pads, like what would be passed to numpy's pad
        function

    Returns
    -------

    slices : list
        A list of slice objects that will be used during buffer creation and
        ghost exchange.
    """

    dim = len(pads)

    slices = [[[None for _a in range(dim)] for _b in range(4)] for _c in range(dim)]

    # For each dimension, we need to compute slice objects for the left bulk,
    # right bulk, left ghost, and right ghost buffers. This requires computing
    # 4*dim slice objects per dim, giving a total of dim*4*dim slice objects.
    # These are organizes into a nested list structure slices[dim][4][dim]
    for (i, _) in _cart_keys[dim]:
        for (j, _) in _cart_keys[dim]:

            # Get pads in the j dimension
            lpad = pads[j][0]
            rpad = pads[j][1]

            if j < i:
                # Slice objects in dimensions "less" than then current dimension
                # do not encompass the full extent of the array
                slices[i][0][j] = slice(lpad, -rpad, 1)
                slices[i][1][j] = slice(lpad, -rpad, 1)
                slices[i][2][j] = slice(lpad, -rpad, 1)
                slices[i][3][j] = slice(lpad, -rpad, 1)

            elif j == i:
                # Slice object in current direction is different depending on
                # which buffer we are indexing for
                slices[i][0][j] = slice(lpad,            lpad + rpad, 1)    # Left bulk
                slices[i][1][j] = slice(-(lpad + rpad), -rpad,        1)    # Right bulk
                slices[i][2][j] = slice(None,            lpad,        1)    # Left ghost
                slices[i][3][j] = slice(-rpad,           None,        1)    # Right ghost

            elif j > i:
                # Slice objects in dimensions "greater" than then current dimension
                # do encompass the full extent of the array
                slices[i][0][j] = slice(None, None, 1)
                slices[i][1][j] = slice(None, None, 1)
                slices[i][2][j] = slice(None, None, 1)
                slices[i][3][j] = slice(None, None, 1)

        slices[i][0] = tuple(slices[i][0])
        slices[i][1] = tuple(slices[i][1])
        slices[i][2] = tuple(slices[i][2])
        slices[i][3] = tuple(slices[i][3])

    return slices

def create_buffers(slices, padded_shape):
    """
    Given an array A with padded shape padded_shape, computes exchange buffers
    to hold copies of the data contained in A for a ghost exchange.

    Parameters
    ----------

    slices : list
        List of slice objects computed from create_slices function

    padded_shape : tuple
        Padded shape of array A

    Returns
    -------

    buffers : list
        List of numpy array objects
    """

    dim = len(padded_shape)

    buffers = [[None for _a in range(4)] for _b in range(dim)]

    for (i, _) in _cart_keys[dim]:

        lbulk_len  = 1
        rbulk_len  = 1
        lghost_len = 1
        rghost_len = 1

        for (j, _) in _cart_keys[dim]:

            # Get precomputed slices
            lbulk_slice = slices[i][0][j]
            rbulk_slice = slices[i][1][j]
            lghost_slice = slices[i][2][j]
            rghost_slice = slices[i][3][j]

            # Get lengths of slices when applied to array with shape
            # padded_shape by using the slice.indices(n) function
            lbulk_len  *= len(range(*lbulk_slice.indices(padded_shape[j])))
            rbulk_len  *= len(range(*rbulk_slice.indices(padded_shape[j])))
            lghost_len *= len(range(*lghost_slice.indices(padded_shape[j])))
            rghost_len *= len(range(*rghost_slice.indices(padded_shape[j])))

        # Set the relevant position in the buffer list to be a zeroed numpy
        # array
        buffers[i][0] = np.zeros(lbulk_len, dtype=float)
        buffers[i][1] = np.zeros(rbulk_len, dtype=float)
        buffers[i][2] = np.zeros(lghost_len, dtype=float)
        buffers[i][3] = np.zeros(rghost_len, dtype=float)

    return buffers

def ghost_exchange(a, slices, buffers, pwrap):
    """
    Performs a ghost exchange on array using precomputed slices and buffers

    Parameters
    ----------

    a : numpy array
        Numpy array padded using the same pads passed to create_slices

    slices : list
        List of slices created by create_slices

    buffers : list
        List of buffers created by create_buffers

    pwrap : ParallelWrapCartesianMesh
        Wrapper for cartesian communicator used in
    """

    dim = len(slices)
    comm = pwrap.cart_comm
    reqs = list()

    for (i, _) in _cart_keys[dim]:

        # Get precomputed buffers
        lbulk_buffer  = buffers[i][0]
        rbulk_buffer  = buffers[i][1]
        lghost_buffer = buffers[i][2]
        rghost_buffer = buffers[i][3]

        # Pack
        lbulk_buffer[:]  = a[slices[i][0]].flatten()[:]
        rbulk_buffer[:]  = a[slices[i][1]].flatten()[:]
        lghost_buffer[:] = a[slices[i][2]].flatten()[:]
        rghost_buffer[:] = a[slices[i][3]].flatten()[:]

        # Send data
        lrank = pwrap.neighbor_ranks[i][0]
        rrank = pwrap.neighbor_ranks[i][1]
        if lrank != -2:
            lbulk_send_req  = comm.Isend(lbulk_buffer, dest=lrank, tag=i)
            lghost_recv_req = comm.Irecv(lghost_buffer, source=lrank, tag=i+dim)
            reqs.append(lbulk_send_req)
            reqs.append(lghost_recv_req)
        if rrank != -2:
            rbulk_send_req  = comm.Isend(rbulk_buffer, dest=rrank, tag=i+dim)
            rghost_recv_req = comm.Irecv(rghost_buffer, source=rrank, tag=i)
            reqs.append(rbulk_send_req)
            reqs.append(rghost_recv_req)

    # Perform communication
    MPI.Request.Waitall(reqs)

    # Unpack data
    for (i, _) in _cart_keys[dim]:

        # Get bulk and ghost region shapes
        lghost_shape = a[slices[i][2]].shape
        rghost_shape = a[slices[i][3]].shape

        # Get bulk and ghost region buffers
        lghost_buffer = buffers[i][2]
        rghost_buffer = buffers[i][3]

        # Copy data
        np.copyto(a[slices[i][2]], lghost_buffer.reshape(lghost_shape))
        np.copyto(a[slices[i][3]], rghost_buffer.reshape(rghost_shape))


if __name__ == '__main__':

    from pysit.util.parallel import ParallelWrapCartesian

    np.set_printoptions(suppress=True, linewidth=200)

    dims = (2, 2, 2)
    pwrap = ParallelWrapCartesian(dims=dims)
    rank = pwrap.cart_rank

    pads = [(1, 2), (1, 2), (1, 2)]
    fill_value = (pwrap.cart_rank + 1) * 10 ** -(pwrap.cart_rank)
    pad_value = (pwrap.cart_rank + 1) * 10

    a = np.full(shape=(5, 5, 5), fill_value=fill_value, dtype=float)
    a = np.pad(a, pad_width=pads, mode='constant', constant_values=pad_value)

    slices = create_slices(pads)
    if rank == 0:
        for i, sl in enumerate(slices):
            dim = _cart_keys[len(pads)][i]
            print(f'dim = {dim}')
            print(f'\tleft bulk = {sl[0]}')
            print(f'\tright bulk = {sl[1]}')
            print(f'\tleft ghost = {sl[2]}')
            print(f'\tright ghost = {sl[3]}')

    buffers = create_buffers(slices, a.shape)
    if rank == 0:
        for i, bf in enumerate(buffers):
            dim = _cart_keys[len(pads)][i]
            print(f'dim = {dim}')
            print(f'\tleft bulk buffer = {bf[0]}')
            print(f'\tright bulk buffer = {bf[1]}')
            print(f'\tleft ghost buffer = {bf[2]}')
            print(f'\tright ghost buffer = {bf[3]}')

    ghost_exchange(a, slices, buffers, pwrap)

    if rank == 0:
        print(a)
