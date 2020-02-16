from mpi4py import MPI

from pysit.core.acquisition import equispaced_acquisition
from pysit.core.domain import RectangularDomain, PML
from pysit.core.mesh import CartesianMesh
from pysit.core.receivers import PointReceiver
from pysit.core.shot import Shot
from pysit.core.sources import PointSource
from pysit.core.wave_source import RickerWavelet
from pysit.gallery import horizontal_reflector
from pysit.modeling.data_modeling_org import generate_seismic_data
from pysit.modeling.temporal_modeling import TemporalModeling
from pysit.solvers import ConstantDensityAcousticWave
from pysit.util.parallel import ParallelWrapCartesian

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Create a cartesian decomposition
    dims = (4,)
    pwrap = ParallelWrapCartesian(dims=dims, comm=MPI.COMM_WORLD)

    # Define the domain
    pmlz = PML(0.1, 100, ftype='quadratic')
    zconfig = (0.1, 0.8, pmlz, pmlz)
    d = RectangularDomain(zconfig)

    # Define the mesh
    m = CartesianMesh(d, 301, pwrap=pwrap)

    print(f'rank = {pwrap.cart_rank}, domain zconfig = {m.domain.z}')

    # Generate true wave speed using gallery function
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

    print(f'Set up shot on rank {pwrap.cart_rank}')

    # Define and configure the wave solver
    trange = (0.0,3.0)
    solver = ConstantDensityAcousticWave(m,
                                         formulation='scalar',
                                         kernel_implementation='numpy',
                                         spatial_accuracy_order=2,
                                         trange=trange)

    print('Generating data...')
    wavefields = []
    base_model = solver.ModelParameters(m, {'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    tools = TemporalModeling(solver)
    m0 = solver.ModelParameters(m, {'C': C0})

    fwdret = tools.forward_model(shots[0], m0, return_parameters=['wavefield', 'dWaveOp', 'simdata'])
    dWaveOp0 = fwdret['dWaveOp']
    inc_field = fwdret['wavefield']
    data = fwdret['simdata']

    clim = C.min(),C.max()

    print(len(inc_field))
