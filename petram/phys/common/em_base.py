#
#  common routine for EM Physics Modules
#
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EMPhysModule')

from petram.phys.phys_model import PhysModule

class EMPhysModule(PhysModule):
    def __init__(self, **kwargs):
        super(EMPhysModule, self).__init__(**kwargs)
        self.freq_txt = "1.0e9"
        
    def attribute_set(self, v):
        v = super(EMPhysModule, self).attribute_set(v)
        v["freq_txt"] = "1.0e9"
        return v
        
    def is_complex(self):
        return True
    
    def get_default_ns(self):
        from petram.phys.phys_const import mu0, epsilon0, q0, massu, chargez
        ns = {'mu0': mu0,
              'e0': epsilon0,
              'q0': q0,
              'massu': massu,
              'chargez': chargez}
        return ns

    def get_freq_omega(self):
        freq, _void = self.eval_param_expr(self.freq_txt, "freq")
        try:
            _void = float(freq)
        except:
            freq = 1e6
            dprint1("Error, frequency must be a scalr real value")
        return freq, 2.*np.pi*freq

    def panel1_param(self):
        panels = super(EMPhysModule, self).panel1_param()
        panels.extend([self.make_param_panel('freq',  self.freq_txt),])

        return panels
    

    def get_panel1_value(self):    
        val = super(EMPhysModule, self).get_panel1_value()
        val.extend([self.freq_txt,])
        
        return val

    def import_panel1_value(self, v):
        v = super(EMPhysModule, self).import_panel1_value(v)
        self.freq_txt = str(v[0])
        return v[1:]
    
