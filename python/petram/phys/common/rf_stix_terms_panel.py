import os
import wx
from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED

from petram.phys.common.rf_dispersion_coldplasma import (stix_options,
                                                         default_stix_option,
                                                         value2panelvalue,
                                                         panelvalue2value)

eps_txt = "\u03B5"
mu_txt = "\u03BC"


def elp_setting(num_ions):
    from petram.phys.common.rf_dispersion_coldplasma import stix_options

    names = ["electrons"]
    for i in range(num_ions):
        names.append("ions"+str(i+1))

    panels = []
    for n in names:
        panels.append([n, None, 36, {"col": 5,
                                     "labels": stix_options}])
    panels.append(
        [None, True, 3, {"text": "include eye(3) for "+eps_txt + " and " + mu_txt}])
    return panels


class dlg_rf_stix_terms(wx.Dialog):
    def __init__(self, parent, num_ions, value):

        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Stix term config",
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        self.SetSizer(vbox)
        vbox.Add(hbox2, 1, wx.EXPAND | wx.ALL, 1)

        ll = elp_setting(num_ions)
        self.elp = EditListPanel(self, ll)

        hbox2.Add(self.elp, 1, wx.EXPAND | wx.RIGHT | wx.LEFT, 1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(self, wx.ID_ANY, "Cancel")
        button2 = wx.Button(self, wx.ID_ANY, "Apply")

        hbox.Add(button, 0, wx.EXPAND)
        hbox.AddStretchSpacer()
        hbox.Add(button2, 0, wx.EXPAND)
        vbox.Add(hbox, 0, wx.EXPAND | wx.ALL, 5)

        button.Bind(wx.EVT_BUTTON, self.onCancel)
        button2.Bind(wx.EVT_BUTTON, self.onApply)

        self.elp.SetValue(value)
        #self.SetSizeHints(minH=-1, minW=size.GetWidth())
        #self.SetSizeHints(minH=-1, minW=300)
        self.Layout()
        self.Fit()
        self.CenterOnParent()

        self.Show()

    def get_value(self):
        return self.elp.GetValue()

    def onCancel(self, evt=None):
        self.value = self.elp.GetValue()
        self.EndModal(wx.ID_CANCEL)

    def onApply(self, evt):
        self.value = self.elp.GetValue()
        self.EndModal(wx.ID_OK)
        evt.Skip()


def ask_rf_stix_terms(win, num_ions, value):
    panelvalue = value2panelvalue(num_ions, value)
    dlg = dlg_rf_stix_terms(win, num_ions, panelvalue)

    try:
        if dlg.ShowModal() == wx.ID_OK:
            panelvalue = dlg.get_value()
            value = panelvalue2value(panelvalue)
        else:
            pass
    finally:
        dlg.Destroy()

    return value
