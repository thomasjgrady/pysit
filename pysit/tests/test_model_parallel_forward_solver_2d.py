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

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Create a cartesian decomposition
    dims = (3, 3)
    pwrap = ParallelWrapCartesian(dims=dims, comm=MPI.COMM_WORLD)

    # Define the domain
    pml = PML(0.1, 100, ftype='quadratic')
    xconfig = (0.1, 0.8, pml, pml)
    zconfig = (0.1, 0.8, pml, pml)
    d = RectangularDomain(xconfig, zconfig)

    # Define the mesh
    m = CartesianMesh(d, 91, 71, pwrap=pwrap)

    # Generate true wave speed using gallery function
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0,3.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    wavefields =  []
    base_model = solver.ModelParameters(m,{'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)

    tools = TemporalModeling(solver)
    m0 = solver.ModelParameters(m, {'C': C0})

    fwdret = tools.forward_model(shots[0], m0, return_parameters=['wavefield', 'dWaveOp', 'simdata'])
    dWaveOp0 = fwdret['dWaveOp']
    inc_field = fwdret['wavefield']
    data = fwdret['simdata']

    inc_field_shapes = pwrap.cart_comm.gather(m.shape(as_grid=True, include_bc=False), root=0)
    inc_fields = pwrap.cart_comm.gather(inc_field, root=0)

    if pwrap.cart_rank == 0:
        clim = C.min(),C.max()
        fields = list()
        for inc_field_list in zip(*inc_fields):
            inc_field_arrs = list()
            for i, f in enumerate(inc_field_list):
                inc_field_arrs.append(np.reshape(np.array(f), newshape=inc_field_shapes[i]))
            
            r1 = np.concatenate(inc_field_arrs[:3],  axis=1)
            r2 = np.concatenate(inc_field_arrs[3:6], axis=1)
            r3 = np.concatenate(inc_field_arrs[6:],  axis=1)
            
            fields.append(np.concatenate([r1, r2, r3], axis=0))

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(fields[-1], cmap='plasma')
        plt.show()
