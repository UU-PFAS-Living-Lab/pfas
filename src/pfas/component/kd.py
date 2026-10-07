
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

class FreundlichSPsorption(BaseModel, validate_assignment=True):
    """Calculate solid-phase Kd using a Freundlich isotherm.

    The Freundlich relationship is evaluated at the representative
    concentration:

        Kd = K_freund * C_rep**(n_freund - 1)

    Parameters
    ----------
    sorption_solid : dict
        Freundlich sorption configuration.

    Returns
    -------
    dict
        Contains ``Kd`` as either a float or a Pint quantity, depending
        on the inputs.
    """

    sorption_solid: dict

    def compute(self):
        cfg = self.sorption_solid.get("freundlich")

        if not cfg:
            raise ValueError(
                "sorption_isotherm is 'freundlich' but the "
                "'freundlich' configuration is missing."
            )

        required_keys = (
            "K_freund",
            "n_freund",
            "C_rep",
        )

        missing_keys = [
            key for key in required_keys if key not in cfg
        ]

        if missing_keys:
            raise ValueError(
                "Freundlich isotherm requires "
                f"{missing_keys} inside sorption_solid['freundlich']."
            )

        Kd = kd_freundlich(
            C_rep=cfg["C_rep"],
            K_freund=cfg["K_freund"],
            n_freund=cfg["n_freund"],
        )

        return {"Kd": Kd}
