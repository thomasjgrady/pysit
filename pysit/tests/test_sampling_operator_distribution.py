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
from pysit.util.communication import create_slices
from pysit.util.parallel import ParallelWrapCartesian

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Create a cartesian decomposition
    dims = (2, 2)
    pwrap = ParallelWrapCartesian(dims=dims, comm=MPI.COMM_WORLD)

    # Define the domain
    pml = PML(0.1, 100, ftype='quadratic')
    xconfig = (0.1, 0.8, pml, pml)
    zconfig = (0.1, 0.8, pml, pml)
    d = RectangularDomain(xconfig, zconfig)
    
    if pwrap.cart_rank == 0: 
        # Define the mesh
        m = CartesianMesh(d, 91, 71)

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

        plt.figure()
        plt.imshow(np.reshape(C, m.shape(as_grid=True)))
        plt.show()

    else:
        C, C0 = None

    C_global  = MPI.bcast(C,  root=0)
    C0_global = MPI.bcast(C0, root=0)

    m = CartesianMesh(d, 91, 71, pwrap=pwrap)
    

