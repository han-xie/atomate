# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines the thermal conductivity workflow
Current output (20170913): FORCE_CONSTANTS_2ND and FORCE_CONSTANTS_3RD
They can be directly used to calculate thermal conductivity
"""

from datetime import datetime
from fireworks import Firework, Workflow

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import OptimizeFW, DispersionFW, ThirdOrderFW
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.parse_outputs import DispersionAnalysisTask, ThermalConductivityTask

from pymatgen.io.vasp.sets import MPRelaxSet

__author__ = 'Han Xie'
__email__ = 'xhyglh@sjtu.edu.cn'

logger = get_logger(__name__)

def get_wf_thermal_conductivity(structure, vasp_input_set=None, mode="force", vasp_cmd="vasp",
                                third_cmd="thirdorder_vasp.py", db_file=None,
                                supercell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]], cutoff_3rd=-4,
                                user_kpoints_settings=None, optimize_structure=True):
    """
    Returns the workflow of calculating thermal conductivity.
    Note: thirdorder package is required.

    Args:
        structure (Structure): input structure.
        vasp_input_set (VaspInputSet): input set to use.
        mode (str): options:
            "force": calculate dispersion from forces.
            "hessian" calculate dispersion from hessian.
        vasp_cmd (str): vasp command to run.
        third_cmd (str): Command to run thirdorder package.
        db_file (str): path to the db file.
        supercell (tuple): supercell size to use.
        cutoff_3rd (+float or -integer): cutoff distance for 3rd order FCs.
        user_kpoints_settings (dict): example: {"grid_density": 7000}
        optimize_stucture (bool): add OptimizeFW or not.

    Returns:
        Workflow
    """
    try:
        from phonopy import Phonopy
    except ImportError:
        logger.warn("The required 'phonopy' package is NOT installed.")

    fws, parent1 = [], []
    vis_orig = vasp_input_set or MPRelaxSet(structure, force_gamma=True)
    uis_common = {"EDIFF": 1E-08}
    tag = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')

    if optimize_structure:
        vis_relax_dict = vis_orig.as_dict()
        uis_relax = vis_relax_dict.get("user_incar_settings", {})
        uis_relax.update(uis_common)
        vis_relax_dict.update({"user_incar_settings": uis_relax})
        if user_kpoints_settings:
            vis_relax_dict.update({"user_kpoints_settings": user_kpoints_settings})
        vis_relax = vis_orig.__class__.from_dict(vis_relax_dict)
        fw1 = OptimizeFW(structure=structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                         db_file=db_file, name="{} structure optimization".format(tag))
        fws = [fw1]
        parent1 = [fw1]

    vis_disp_dict = vis_orig.as_dict()
    if user_kpoints_settings:
        vis_disp_dict.update({"user_kpoints_settings": user_kpoints_settings})
    uis_disp = vis_disp_dict.get("user_incar_settings", {})
    uis_disp.update(uis_common)

    parent2 = []
    if mode == "force":
        uis_disp.update({'IBRION': 2, 'ISMEAR': 0, 'ISPIN': 1, 'NSW': 0})
        vis_disp_dict.update({"user_incar_settings": uis_disp})
        vis_disp = vis_orig.__class__.from_dict(vis_disp_dict)
        fw2 = DispersionFW(structure=structure, vasp_input_set=vis_disp, vasp_cmd=vasp_cmd,
                           db_file=db_file, mode=mode, parents=parent1, supercell=supercell,
                           name="{} dispersion".format(tag))
        fws.append(fw2)
        fw3 = Firework([PassCalcLocs(name="{}-{} dispersion".
                        format(structure.composition.reduced_formula, tag)),
                        DispersionAnalysisTask(tag=tag, db_file=db_file, mode=mode, supercell=supercell)],
                        parents=[fw2], name="{}-{} dispersion".
                        format(structure.composition.reduced_formula, tag))
        fws.append(fw3)
        parent2 = [fw3]
#        wf_disp = Workflow(fws)
    elif mode == "hessian":
        uis_disp.update({'IBRION': 6, 'ISMEAR': 0, 'ISPIN': 1, 'NSW': 1, 'POTIM': 0.015})
        vis_disp_dict.update({"user_incar_settings": uis_disp})
        vis_disp = vis_orig.__class__.from_dict(vis_disp_dict)
        fw2 = DispersionFW(structure=structure, vasp_input_set=vis_disp, vasp_cmd=vasp_cmd,
                           db_file=db_file, mode=mode, parents=parent1, supercell=supercell,
                           name="{} dispersion".format(tag))
        fws.append(fw2)
        parent2 = [fw2]
#        wf_disp = Workflow(fws)
    else:
        raise ValueError('Dispersion workflow mode should be "force" or "hessian".')

#   Start to run thirdorder
    vis_third_dict = vis_orig.as_dict()
    if user_kpoints_settings:
        vis_third_dict.update({"user_kpoints_settings": user_kpoints_settings})
    uis_third = vis_third_dict.get("user_incar_settings", {})
    uis_third.update(uis_common)
    uis_third.update({'IBRION': 2, 'ISMEAR': 0, 'ISPIN': 1, 'NSW': 0})
    vis_third_dict.update({"user_incar_settings": uis_third})
    vis_third = vis_orig.__class__.from_dict(vis_third_dict)
    fw4 = ThirdOrderFW(structure=structure, vasp_input_set=vis_third, third_cmd=third_cmd,
                       vasp_cmd=vasp_cmd, db_file=db_file, parents=parent1,
                       supercell=supercell, cutoff=cutoff_3rd, name="{} thirdorder".format(tag))
    fws.append(fw4)
    parent2.append(fw4)
    fw5 = Firework(ThermalConductivityTask(tag=tag, db_file=db_file, third_cmd=third_cmd,
                   supercell=supercell, cutoff=cutoff_3rd), parents=parent2,
                   name="{}-thermal conductivity".format(structure.composition.reduced_formula))
    fws.append(fw5)
    wf_k = Workflow(fws)
    wf_k.name = "{}-{}".format(structure.composition.reduced_formula, "thermal conductivity")

    return wf_k
