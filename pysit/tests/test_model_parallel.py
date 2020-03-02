from mpi4py import MPI
from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.util.parallel import ParallelWrapCartesian

import numpy as np
import matplotlib.pyplot as plt
import time

MODEL_PARALLEL = True

def test_full_solver(test_num):
    
    # Define a parallel decomposition
    dims = (2, 2)
    pwrap = ParallelWrapCartesian(dims=dims, comm=MPI.COMM_WORLD)
    rank = pwrap.cart_rank
    size = pwrap.size

    # Create a global domain and mesh. This is a hack and would not be included
    # in the final version of any codes, but makes creating the sampling operators
    # correctly less difficult
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)
    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)
    
    nx = 101
    nz = 101
    d_global = RectangularDomain(x_config, z_config)
    m_global = CartesianMesh(d_global, nx, nz)


    if MODEL_PARALLEL:
        # Set up local mesh
        m = CartesianMesh(d_global, nx, nz, pwrap=pwrap)

        #   Generate true wave speed in the global domain
        C_global, C0_global, m_global, d_global = horizontal_reflector(m_global)
        
        # Get the relevant section of the true wave speed and initial guess on this
        # rank
        ns, os = pwrap.split_discrete([nx, nz])
        print(f'test = {test_num}, rank = {rank}, ns = {ns}, os = {os}')
        print(f'rank = {rank}, shape = {m.shape(as_grid=True, include_bc=False)}')
        
        sls = list()
        for n, o in zip(ns, os):
            sls.append(slice(o, o + n, 1))
        sls = tuple(sls)

        C  = np.reshape(C_global,  m_global.shape(as_grid=True, include_bc=False))[sls]
        C0 = np.reshape(C0_global, m_global.shape(as_grid=True, include_bc=False))[sls]
        C  = np.reshape(C,  m.shape(as_grid=False, include_bc=False))
        C0 = np.reshape(C0, m.shape(as_grid=False, include_bc=False))

    else:
        C = C_global
        C0 = C0_global
        m = m_global
        d = d_global

    # Set up shots
    zmin = d_global.z.lbound
    zmax = d_global.z.rbound
    zpos = zmin + (1./9.)*zmax

    n_receivers = 10
    source = PointSource(m, (zpos, zpos), RickerWavelet(25.0))
    receivers = ReceiverSet(m, [PointReceiver(m, (x, zpos)) for x in np.linspace(0.1, 1.0, n_receivers)])
    shots = [Shot(source, receivers)]

    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityAcousticWave(m,
                                         formulation='scalar',
                                         spatial_accuracy_order=2,
                                         trange=trange)

    # Generate synthetic Seismic data
    print(f'rank = {rank}, generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    wavefields1 =  []
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields1)
    print(f'rank = {rank}, data generation took {(time.time()-tt)} s')

    # Define and configure the objective function
    objective = TemporalLeastSquares(solver)

    # Define the inversion algorithm
    invalg = GradientDescent(objective)
    
    # invalg = GradientDescent(objective)
    initial_value = solver.ModelParameters(m, {'C': C0}, padded=False)

    # Execute inversion algorithm
    print(f'rank = {rank}, running descent...')
    tt = time.time()

    nsteps = 20

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

    invalg.max_linesearch_iterations=10

    result = invalg(shots, initial_value, nsteps, verbose=True, status_configuration=status_configuration)

    print(f'rank = {rank}, run time {time.time()-tt}')

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    result_model = np.reshape(result.C, m.shape(as_grid=True, include_bc=False))

    title_string = 'Inversion result'
    if MODEL_PARALLEL:
        title_string += f' on rank {rank}'

    plt.figure()
    plt.title(title_string)
    plt.imshow(result_model)
    plt.show()


if __name__ == '__main__':
    
    test = 0
    test_full_solver(test)
    test += 1
    