import matplotlib.pyplot as plt
import numpy as np
import time

from mpi4py import MPI
from pysit import *
from pysit.util.parallel import ParallelWrapCartesian
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':

    # Define decomposition
    dims = (4,)
    pwrap = ParallelWrapCartesian(dims=dims, comm=MPI.COMM_WORLD)
    rank = pwrap.cart_rank
    size = pwrap.size

    #   Define Domain
    pmlz = PML(0.1, 100, ftype='quadratic')
    z_config = (0.1, 0.8, pmlz, pmlz)
    d = RectangularDomain(z_config)

    # Define mesh
    m = CartesianMesh(d, 101, pwrap=pwrap)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    Nshots = 1
    shots = []

    # Define source location and type
    zpos = 0.2
    source = PointSource(m, (zpos), RickerWavelet(25.0))

    # Define set of receivers
    receiver = PointReceiver(m, (zpos))

    # Create and store the shot
    shot = Shot(source, receiver)
    shots.append(shot)

    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityAcousticWave(m,
                                          kernel_implementation='numpy',
                                          formulation='scalar',
                                          spatial_accuracy_order=2,
                                          trange=trange)

    # Generate synthetic Seismic data
    print(f'Generating data on rank {rank}...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    wavefields = []
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    # Define and configure the objective function
    objective = TemporalLeastSquares(solver)

    invalg = GradientDescent(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running Descent...')
    tt = time.time()

    nsteps = 50

    status_configuration = {'value_frequency'           : 1,
                            'residual_length_frequency' : 1,
                            'objective_frequency'       : 1,
                            'step_frequency'            : 1,
                            'step_length_frequency'     : 1,
                            'gradient_frequency'        : 1,
                            'gradient_length_frequency' : 1,
                            'run_time_frequency'        : 1,
                            'alpha_frequency'           : 1,
                            }

    invalg.max_linesearch_iterations=100

    result = invalg(shots, initial_value, nsteps, verbose=True, status_configuration=status_configuration)

    print('...run time:  {0}s'.format(time.time()-tt))

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])
