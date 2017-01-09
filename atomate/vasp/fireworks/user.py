import os
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from fireworks.core.firework import Firework, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from atomate.vasp.firetasks.parse_outputs import VaspToDbTask


class MPRelaxSetEx(MPRelaxSet):

    def __init__(self, structure, rm_incar_settings=[],
                 user_kpoints_settings={},  **kwargs):
        super(MPRelaxSet, self).__init__(
            structure, MPRelaxSet.CONFIG, **kwargs)
        self.rm_incar_settings = rm_incar_settings
        self.user_kpoints_settings = user_kpoints_settings
        self.rm_config_dict = {"INCAR": {}}
        for k in self.rm_incar_settings:
            if k in self.config_dict["INCAR"].keys():
                self.rm_config_dict["INCAR"][
                    k] = self.config_dict["INCAR"].pop(k, None)

    @property
    def kpoints(self):
        settings = self.user_kpoints_settings or self.config_dict["KPOINTS"]
        if settings.get('kpts'):
            return Kpoints.gamma_automatic(kpts=tuple(settings['kpts']))
        else:
            return super(MPRelaxSet, self).kpoints


@explicit_serialize
class WriteVaspRelaxFromStructure(FiretaskBase):
    """
    Writes input files for a static run. Assumes that output files from a
    relaxation job can be accessed. Also allows lepsilon calcs.
    Required params:
        (none)
    Optional params:
        (documentation for all optional params can be found in
        MPRelaxSetEx)
    """

    required_params = ["struct_dir"]
    optional_params = ["force_gamma",
                       "user_incar_settings", "user_kpoints_settings"]

    def run_task(self, fw_spec):
        struct_dir = self.get("struct_dir", ".")
        struct_file = os.path.join(struct_dir, "POSCAR")
        structure = Structure.from_file(struct_file)
        vis = MPRelaxSetEx(structure,
                           force_gamma=self.get(
                               "force_gamma", True),
                           user_incar_settings=self.get(
                               "user_incar_settings", {}),
                           user_kpoints_settings=self.get(
                               "user_kpoints_settings", None))
        vis.write_input(".")


class OptimizeStepFW(Firework):

    def __init__(self, structure, step_index, name="Relax",
                 vasp_input_set=None, vasp_cmd="vasp",
                 override_default_vasp_params=None, ediffg=None,
                 job_type="normal", db_file=None, parents=None, **kwargs):
        """
        Standard structure optimization Firework.
        Args:
            structure (Structure): Input structure.
            step_index (int): Index of current firework in relax fws list.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use.
                Defaults to MPRelaxSet() if None.
            override_default_vasp_params (dict): If this is not None,
                these params are passed to the default vasp_input_set, i.e.,
                MPRelaxSet. This allows one to easily override some
                settings, e.g., user_incar_settings, etc.
            vasp_cmd (str): Command to run vasp.
            ediffg (float): Shortcut to set ediffg in certain jobs
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework.
                FW or list of FWS.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
#         override_default_vasp_params = override_default_vasp_params
#         job_type = job_type

        t = []

        if step_index:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True))
            t.append(
                WriteVaspRelaxFromStructure(
                    struct_dir='.', force_gamma=True,
                    **override_default_vasp_params)
            )
        else:
            vasp_input_set = vasp_input_set or MPRelaxSetEx(
                structure, force_gamma=True, **override_default_vasp_params)
            t.append(WriteVaspFromIOSet(
                structure=structure, vasp_input_set=vasp_input_set))

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, job_type=job_type,
                                  max_force_threshold=0.25, ediffg=ediffg,
                                  auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDbTask(db_file=db_file,
                              additional_fields={"task_label": name}))
        super(OptimizeStepFW, self).__init__(
            t, parents=parents, name="{}-{}-Step{}".format(
                structure.composition.reduced_formula, name, 1 + step_index
            ), **kwargs)
