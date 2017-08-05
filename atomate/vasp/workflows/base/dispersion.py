# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines the dispersion workflow
1. calculated from forces (FORCE_SETS)
2. calculated from hessian
"""

from datetime import datetime
from fireworks import Firework, Workflow

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import OptimizeFW, DispersionFW
from atomate.vasp.firetasks.parse_outputs import DispersionAnalysisTask

from pymatgen.io.vasp.sets import MPRelaxSet

__author__ = 'Han Xie'
__email__ = 'xhyglh@sjtu.edu.cn'

logger = get_logger(__name__)


def get_wf_dispersion(structure, vasp_input_set=None, vasp_cmd="vasp", db_file=None,
                      mode="force", supercell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                      user_kpoints_settings=None, optimize_structure=True):
    """
    Returns the workflow of calculating dispersion.
    Note: phonopy package is required.

    Args:
        structure (Structure): input structure.
        vasp_input_set (VaspInputSet): input set to use.
        vasp_cmd (str): vasp command to run.
        db_file (str): path to the db file.
        mode (str): options:
            "force": calculate dispersion from forces.
            "hessian" calculate dispersion from hessian.
        supercell (tuple): supercell size to use.
        user_kpoints_settings (dict): example: {"grid_density": 7000}
        optimize_stucture (bool): add OptimizeFW or not.

    Returns:
        Workflow
    """
    try:
        from phonopy import Phonopy
    except ImportError:
        logger.warn("The required 'phonopy' package is NOT installed.")

    fws, parent2 = [], []
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
        if mode == "force":
            fw1 = OptimizeFW(structure=structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                             db_file=db_file, name="{} structure optimization".format(tag))
        else:
            fw1 = OptimizeFW(structure=structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                             db_file=db_file)
        fws = [fw1]
        parent2 = [fw1]

    vis_disp_dict = vis_orig.as_dict()
    if user_kpoints_settings:
        vis_disp_dict.update({"user_kpoints_settings": user_kpoints_settings})
    uis_disp = vis_disp_dict.get("user_incar_settings", {})
    uis_disp.update(uis_common)

    if mode == "force":
        uis_disp.update({'IBRION': 2, 'ISMEAR': 0, 'ISPIN': 1, 'NSW': 0})
        vis_disp_dict.update({"user_incar_settings": uis_disp})
        vis_disp = vis_orig.__class__.from_dict(vis_disp_dict)
        fw2 = DispersionFW(structure=structure, vasp_input_set=vis_disp, vasp_cmd=vasp_cmd,
                           db_file=db_file, mode=mode, parents=parent2, supercell=supercell,
                           name="{} dispersion".format(tag))
        fws.append(fw2)
        fw3 = Firework(DispersionAnalysisTask(tag=tag, db_file=db_file, mode=mode, supercell=supercell),
                       parents=[fw2], name="{}-dispersion_force: analysis".
                       format(structure.composition.reduced_formula))
        fws.append(fw3)
        wf_disp = Workflow(fws)
    elif mode == "hessian":
        uis_disp.update({'IBRION': 6, 'ISMEAR': 0, 'ISPIN': 1, 'NSW': 1, 'POTIM': 0.015})
        vis_disp_dict.update({"user_incar_settings": uis_disp})
        vis_disp = vis_orig.__class__.from_dict(vis_disp_dict)
        fw2 = DispersionFW(structure=structure, vasp_input_set=vis_disp, vasp_cmd=vasp_cmd,
                           db_file=db_file, mode=mode, parents=parent2, supercell=supercell)
        fws.append(fw2)
        wf_disp = Workflow(fws)
    else:
        raise ValueError('Dispersion workflow mode should be "force" or "hessian".')

    wf_disp.name = "{}-{}".format(structure.composition.reduced_formula, "dispersion")

    return wf_disp
