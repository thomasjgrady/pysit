# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.util.parallel import ParallelWrapCartesian

def split_discrete(n_globals, dims, coords):

    # Number of points and point offsets for each dimension
    n_locals = list()
    o_locals = list()

    for n_global, dim, coord in zip(n_globals, dims, coords):
        remainder = n_global % dim
        if coord < remainder:
            n_local = n_global // dim + 1
            o_local = coord * n_local
        else:
            n_local = n_global // dim
            o_local = (n_local + 1) * remainder + (coord - remainder) * n_local
        n_locals.append(n_local)
        o_locals.append(o_local)

    return n_locals, o_locals

if __name__ == '__main__':
    # Setup
    dims = (2, 2)
    pwrap = ParallelWrapCartesian(dims=dims, comm=MPI.COMM_WORLD)
    rank = pwrap.cart_rank
    size = pwrap.size
    coords = pwrap.cart_coords

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    nx = 91
    nz = 71
    m_global = CartesianMesh(d, nx, nz)

    #   Generate true wave speed
    C_global, C0_global, m_global, d = horizontal_reflector(m_global)

    # Distribute true wave speed to ranks 
    n_locals, o_locals = split_discrete([nx, nz], dims, coords)
    sls = tuple([slice(o, o + n, 1) for o, n in zip(o_locals, n_locals)])
    
    C  = np.reshape(C_global, m_global.shape(as_grid=True))[sls]
    C0 = np.reshape(C0_global, m_global.shape(as_grid=True))[sls]
    
    # Construct a local mesh
    m = CartesianMesh(d, nx, nz, pwrap=pwrap)

    C = np.reshape(C, m.shape())
    C0 = np.reshape(C0, m.shape())

    # Make shots
    zpos = 0.2
    source = PointSource(m, (zpos, zpos), RickerWavelet(10.0))
    receivers = ReceiverSet(m, [PointReceiver(m, (xpos, zpos)) for xpos in np.linspace(0.1, 1.0, nx // 10)])
    shots = [Shot(source, receivers)]

    # Define and configure the wave solver
    trange = (0.0, 3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    base_model = solver.ModelParameters(m,{'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    print('Data generation: {0}s'.format(time.time()-tt))

    objective = TemporalLeastSquares(solver)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()

    nsteps = 5

    status_configuration = {'value_frequency'           : 1,
                            'residual_frequency'        : 1,
                            'residual_length_frequency' : 1,
                            'objective_frequency'       : 1,
                            'step_frequency'            : 1,
                            'step_length_frequency'     : 1,
                            'gradient_frequency'        : 1,
                            'gradient_length_frequency' : 1,
                            'run_time_frequency'        : 1,
                            'alpha_frequency'           : 1,
                            }

#   line_search = ('constant', 1e-16)
    line_search = ('constant', 1e-8)

    result = invalg(shots, initial_value, nsteps,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True)

    print('rank {0} run time: {1}s'.format(rank, time.time()-tt))

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.title(f'{rank}')
    plt.imshow(np.reshape(result.C, m.shape(as_grid=True)))
    plt.show()
