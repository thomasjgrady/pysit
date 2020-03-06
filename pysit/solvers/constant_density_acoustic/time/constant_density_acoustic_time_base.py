

import math

import numpy as np

from ..constant_density_acoustic_base import *

from pysit.util.solvers import inherit_dict
from pysit.util.communication import create_slices, create_buffers, ghost_exchange

try:
    from mpi4py import MPI
except ImportError:
    pass

__all__ = ['ConstantDensityAcousticTimeBase']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeBase(ConstantDensityAcousticBase):

    _local_support_spec = {'equation_dynamics': 'time',
                           # These should be defined by subclasses.
                           'temporal_integrator': None,
                           'temporal_accuracy_order': None,
                           'spatial_discretization': None,
                           'spatial_accuracy_order': None,
                           'kernel_implementation': None,
                           'spatial_dimension': None,
                           'boundary_conditions': None,
                           'precision': None}

    def __init__(self,
                 mesh,
                 trange=(0.0, 1.0),
                 cfl_safety=1/6,
                 **kwargs):

       
        self.trange = trange
        self.cfl_safety = cfl_safety

        self.t0, self.tf = trange
        self.dt = 0.0
        self.nsteps = 0

        ConstantDensityAcousticBase.__init__(self,
                                             mesh,
                                             trange=trange,
                                             cfl_safety=cfl_safety,
                                             **kwargs)

        if self.mesh.pwrap.size > 1:
            self.pads = [(1, 1)] * self.mesh.dim
            self.slices = create_slices(self.pads)
            self.buffers = create_buffers(self.slices, self.mesh.shape(as_grid=True, include_bc=True))

    def ts(self):
        """Returns a numpy array of the time values serviced by the specified dt
        and trange."""
        return np.arange(self.nsteps)*self.dt

    def _process_mp_reset(self, *args, **kwargs):

        if self.mesh.pwrap.size > 1:
            mp_reshaped = np.reshape(self._mp.C.data, self.mesh.shape(as_grid=True, include_bc=True))
            ghost_exchange(mp_reshaped, self.slices, self.buffers, self.mesh.pwrap)
            self._mp.C.data = np.reshape(mp_reshaped, self.mesh.shape(as_grid=False, include_bc=True))

        CFL = self.cfl_safety
        t0, tf = self.trange

        min_deltas = np.min(self.mesh.deltas)

        C = self._mp.C
        max_C = max(abs(C.min()), C.max())  # faster than C.abs().max()

        dt = CFL*min_deltas / max_C
        nsteps = int(math.ceil((tf - t0)/dt))
        

        # Here, in the case of a parallel mesh, the number of steps will
        # sometimes be different between ranks. Thus, if we have more than one
        # worker we need to do a reduction to make nsteps the max across all
        # ranks, thus ensuring that the CFL safety condition holds for each
        # rank.
        if self.mesh.type == 'structured-cartesian':
            if self.mesh.pwrap.size > 1:
                nsteps = self.mesh.pwrap.comm.allreduce(nsteps, op=MPI.MAX)

        self.dt = dt
        self.nsteps = nsteps

        self._rebuild_operators()

    def _rebuild_operators(self, *args, **kwargs):
        pass

    def time_step(self, solver_data, rhs, **kwargs):
        raise NotImplementedError("Function 'time_step' Must be implemented by subclass.")
