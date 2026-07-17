
"""Solid-phase sorption preprocessing components."""

from pydantic import BaseModel

from pfas.utils import kd_fabregat_palau, kd_freundlich


class LinearSPsorption(BaseModel, validate_assignment=True, extra='forbid'):
    """Calculate the retardation factor for sorption to the solid phase.

    Three Kd resolution methods are supported, selected via the
    ``sorption_isotherm`` and ``Kd_method`` keys inside *sorption_solid*:

    **Linear isotherm — direct input** (``sorption_isotherm: "linear"``,
    ``Kd_method: "direct_input"``)
        The distribution coefficient is supplied directly::

            sorption_solid = {
                "sorption_isotherm": "linear",
                "linear": {"Kd_method": "direct_input", "Kd": 0.042},
                ...
            }

    **Linear isotherm — Fabregat-Palau (2021)** (``sorption_isotherm: "linear"``,
    ``Kd_method: "fabregat_palau"``)
        Kd is estimated from molecular structure and soil composition using
        :func:`pfas.utils.kd_fabregat_palau`::

            sorption_solid = {
                "sorption_isotherm": "linear",
                "linear": {
                    "Kd_method": "fabregat_palau",
                    "n_CFx": 7,
                    "f_oc": 0.0004,
                    "f_silt_clay": 0.0,
                },
                ...
            }

    Parameters
    ----------
    sorption_solid : dict
        Dictionary containing sorption parameters as described above.

    Attributes
    ----------
    outputs : list of str
        List containing ``'Kd'``.
    """
    sorption_solid: dict

    def compute(self):
        cfg = self.sorption_solid.get("linear")
        if not cfg:
            raise ValueError(
                "sorption_isotherm is 'linear' but 'linear' key is missing "
                "from sorption_solid."
            )

        kd_method = cfg.get("Kd_method", "direct_input")

        if kd_method == "direct_input":
            if "Kd" not in cfg:
                raise ValueError(
                    "Kd_method 'direct_input' requires 'Kd' "
                    "inside sorption_solid['linear']."
                )
            kd = cfg["Kd"]

        elif kd_method == "fabregat_palau":
            for key in ("n_CFx", "f_oc", "f_silt_clay"):
                if key not in cfg:
                    raise ValueError(
                        f"Kd_method 'fabregat_palau' requires '{key}' "
                        "inside sorption_solid['linear']."
                    )
            kd = kd_fabregat_palau(cfg["n_CFx"], cfg["f_oc"], cfg["f_silt_clay"])

        else:
            raise ValueError(
                f"Unsupported Kd_method '{kd_method}' for linear isotherm. "
                "Choose from: 'direct_input', 'fabregat_palau'."
            )

        return {"Kd": kd}

class FreundlichSPsorption(BaseModel, validate_assignment=True, extra='forbid'):
    """Calculate the solid-phase retardation factor using a Freundlich isotherm.

    Parameters
    ----------
    sorption_solid : dict
        Dictionary containing sorption parameters as described above.
    bulk_density : float
        Soil bulk density (kg/m³). Must be positive.
    hydro_properties : HydrologicalProperties
        Hydraulic properties from WaterPreprocessor.

    Attributes
    ----------
    outputs : list of str
        List containing ``'Kd'``.
    """
    sorption_solid: dict
    def compute(self):

        cfg = self.sorption_solid.get("freundlich")
        if not cfg:
            raise ValueError(
                "sorption_isotherm is 'freundlich' but 'freundlich' key is missing "
                "from sorption_solid."
            )
        for key in ("K_freund", "n_freund"):
            if key not in cfg:
                raise ValueError(
                    f"Freundlich isotherm requires '{key}' "
                    "inside sorption_solid['freundlich']."
                )
        C_rep = cfg.get("C_rep", 1.0)  # noqa: N806
        kd = kd_freundlich(C_rep, cfg["K_freund"], cfg["n_freund"])
        return {"Kd": kd}
