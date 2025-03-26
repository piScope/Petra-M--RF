from petram.mfem_config import use_parallel
import os
import numpy as np

from petram.solver.parametric_scanner import DefaultParametricScanner
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('PortScanner')
dprint0 = debug.regular_print('PortScanner', True)
format_memory_usage = debug.format_memory_usage

if use_parallel:
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    smyid = '.'+'{:0>6d}'.format(myid)
    from mfem.common.mpi_debug import nicePrint
else:
    myid = 0
    smyid = ''


class PortScanner(DefaultParametricScanner):
    ''' 
    Scanner for port BC amplitude

    PortScan([1,2,3], amplitude=1, phase=0.0, phys='first')
    PortScan([1,2,3], amplitude=1, phase=0.0, phys='EM3D1')
    PortScan(amplitude=1, phase=0.0)  " scan all port of first phys em model"

    # user need to load this in the global namelist

    '''

    def __init__(self, *args, **kwargs):
        self.amplitude = kwargs.pop("amplitude", 1)
        self.phase = kwargs.pop("phase", 0.0)
        self.phys_name = kwargs.pop("phys", 'first')

        if len(args) == 0:
            self._names = None
            self.port = 'all'
            data = []
        else:
            self.port = [str(x) for x in args[0]]
            data = list(range(len(self.port)))
            names = [str(x) for x in self.port]
            self._names = ["ports_"+"_".join(names)]

        self.smat = None
        DefaultParametricScanner.__init__(self, data=data)

    def get_probes(self):
        return ["Smat"]

    def set_data_from_model(self, root):
        from petram.phys.em3d.em3d_port import EM3D_Port
        from petram.phys.em3d.em3d_portarray import EM3D_PortArray
        #from petram.phys.em2d.em2d_port import EM2D_Port
        from petram.phys.em2da.em2da_port import EM2Da_Port
        from petram.phys.em1d.em1d_port import EM1D_Port

        phys_root = root['Phys']

        if self._names is None:  # automatic scan of all ports in first EM model
            iports = []
            for phys in phys_root:
                if (self.phys_name != 'first' and
                        not phys != self.phys_name):
                    continue
                found = False
                for obj in phys_root[phys].walk():
                    if not obj.enabled:
                        continue
                    if (isinstance(obj, EM3D_Port) or
                        isinstance(obj, EM2Da_Port) or
                            isinstance(obj, EM1D_Port)):

                        iports.append(str(obj.port_idx))
                        found = True
                    elif isinstance(obj, EM3D_PortArray):
                        for i in range(len(obj.sel_index)):
                            iports.append(str(obj.port_idx) + '_' + str(i+1))
                        found = True
                if found:
                    self.phys_name = phys
                    dprint1("Scanned all ports in " + phys)
                    break
            self.port = iports
            names = ["port."+str(x) for x in self.port]
            data = list(range(len(self.port)))

            self.set_data(data)

        elif self.phys_name == 'first':  # set phys_name to first EM model
            for phys in phys_root:
                found = False
                for obj in phys_root[phys].walk():
                    if not obj.enabled:
                        continue
                    if (isinstance(obj, EM3D_Port) or
                        isinstance(obj, EM2Da_Port) or
                        isinstance(obj, EM1D_Port) or
                            isinstance(obj, EM3D_PortArray)):
                        found = True
                if found:
                    self.phys_name = phys
                    break
        dprint1("Port scanner is used for :" + self.phys_name)

    def apply_param(self, data):
        from petram.phys.em3d.em3d_port import EM3D_Port
        from petram.phys.em3d.em3d_portarray import EM3D_PortArray
        #from petram.phys.em2d.em2d_port import EM2D_Port
        from petram.phys.em2da.em2da_port import EM2Da_Port
        from petram.phys.em1d.em1d_port import EM1D_Port

        dprint1("Port Scanner: Port Amplitude: " + str(data))

        for phys in self.target_phys:
            if phys.name() != self.phys_name:
                continue
            for obj in phys.walk():
                if not obj.enabled:
                    continue

                if (isinstance(obj, EM3D_Port) or
                    isinstance(obj, EM2Da_Port) or
                        isinstance(obj, EM1D_Port)):

                    obj.inc_amp_txt = '0.0'
                    if self.port[data] == str(obj.port_idx):
                        obj.inc_amp_txt = str(self.amplitude)
                        obj.inc_phase_txt = str(self.phase)
                        dprint1("Port Scanner: Setting port:",
                                obj, "=", obj.inc_amp_txt)

                elif isinstance(obj, EM3D_PortArray):
                    obj.inc_phase_txt = str(self.phase)
                    dd = [0.0]*len(obj.sel_index)
                    for i in range(len(obj.sel_index)):
                        n = str(obj.port_idx) + '_' + str(i+1)
                        if self.port[data] == n:
                            dd[i] = self.amplitude
                    obj.inc_amp_txt = ','.join([str(x) for x in dd])
                    dprint1("Port Scanner: Setting port:",
                            obj, "=", obj.inc_amp_txt)

    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        return self._names

    def collect_probe_signals(self, engine, dirs):
        from petram.phys.em3d.em3d_port import EM3D_Port
        from petram.phys.em3d.em3d_portarray import EM3D_PortArray
        #from petram.phys.em2d.em2d_port import EM2D_Port
        from petram.phys.em2da.em2da_port import EM2Da_Port
        from petram.phys.em1d.em1d_port import EM1D_Port

        pnames = []
        for phys in self.target_phys:
            if phys.name() != self.phys_name:
                continue

            file_suffix = phys.dep_vars_suffix
            for obj in phys.walk():
                if (isinstance(obj, EM3D_Port) or
                    isinstance(obj, EM2Da_Port) or
                        isinstance(obj, EM1D_Port)):
                    # this assumes port BC define only 1 probe signal, which
                    # is true at this moment
                    pnames.append((int(obj.port_idx), obj.get_probes()[0]))

                elif isinstance(obj, EM3D_PortArray):
                    dprint1("Smatrix collection is not supported for PortArray")

        ports = [x[0] for x in sorted(pnames)]
        pnames = [x[1] for x in sorted(pnames)]

        from petram.sol.probe import list_probes, load_probe,  Probe

        od = os.getcwd()
        filenames, probenames = list_probes(od)

        files = dict(zip(probenames, filenames))
        smat = []
        for p in pnames:
            if p+smyid not in files:
                # myid != 0 does not have data.
                continue
            f = files[p+smyid]
            xdata, ydata = load_probe(f)
            smat.append(ydata)

        if len(smat) != 0:
            p = Probe("Smat"+file_suffix, xnames=["ports"])
            for idx, item in zip(ports, smat):
                p.append_value(item, t=idx)
            p.write_file()

            # store Smatrix to _variables
            smat = np.vstack(smat)
            from petram.helper.variables import Constant
            engine.model._variables["Smat"+file_suffix] = Constant(smat)

        if use_parallel:
            from petram.helper.mpi_recipes import scatter_vector
            if myid == 0:
                s = np.array(smat.shape, dtype=np.int32)
                tmp = smat.flatten().astype(np.complex128)
            else:
                s = None
                tmp = None

            s = MPI.COMM_WORLD.bcast(s)
            tmp = MPI.COMM_WORLD.bcast(tmp)
            smat = tmp.reshape(s)

        # store this so that we can access this from sanner in global_ns
        self.Smat = smat


PortScan = PortScanner
