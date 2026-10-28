import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # PFAS soil simulation

        Use the **sidebar** to pick one or more PFAS, a Staring soil, an
        air–water interfacial area (Aaw) method, the grid depth, target depth,
        total simulation time, average infiltration rate and the pulse
        (concentration and duration), each with the unit of your choice, then click **Run simulation**. Results appear
        below in three tabs: *Overview*, *Breakthrough curves* and
        *Details per PFAS*.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    from pfas.data_loader import load_dataset
    from matplotlib import pyplot as plt

    pfas_db = load_dataset("PFASs")
    soil_db = load_dataset("soils")
    preferred_pfas_order = [
        "TFA", "PFBA", "HFPO-DA", "PFPeA", "PFBS", "PFHxA",
        "PFHpA", "PFHxS", "PFOA", "PFNA", "PFOS", "PFDA",
    ]
    pfas_names = [name for name in preferred_pfas_order if name in pfas_db]
    pfas_names.extend(sorted(name for name in pfas_db if name not in pfas_names))
    staring_soils = sorted(
        name for name in soil_db if name.startswith("Staring-O")
    )

    if not pfas_names:
        raise ValueError("No PFAS entries were found in the PFASs dataset.")
    if not staring_soils:
        raise ValueError("No soils starting with 'Staring-O' were found.")

    return pfas_db, pfas_names, plt, soil_db, staring_soils


@app.cell(hide_code=True)
def _(mo, pfas_names, staring_soils):
    aaw_method_options = {
        "Thermodynamic (SWC sorption)": "thermo",
        "GSSA-based": "func_GSSA",
        "d50-based": "func_d50",
        "d50-based, nonlinear saturation": "func_nonlin_d50",
    }
    # Unit options: label shown to the user -> unit string understood by pint.
    depth_unit_options = {"cm": "centimeter", "mm": "millimeter", "m": "meter"}
    time_unit_options = {
        "years": "year",
        "months": "month",
        "days": "day",
        "seconds": "second",
    }
    infiltration_unit_options = {
        "cm/s": "cm/s",
        "mm/day": "mm/day",
        "cm/day": "cm/day",
        "mm/year": "mm/year",
        "cm/year": "cm/year",
        "m/year": "m/year",
    }

    # Pulse concentration: molar units are multiplied by each PFAS's molar mass,
    # mass units are applied as-is to every selected PFAS.
    pulse_concentration_unit_options = {
        "nmol/L": "nanomole/L",
        "µmol/L": "micromole/L",
        "mol/L": "mole/L",
        "ng/L": "nanogram/L",
        "µg/L": "microgram/L",
        "mg/L": "milligram/L",
    }
    pulse_duration_unit_options = {
        "years": "year",
        "months": "month",
        "days": "day",
        "seconds": "second",
    }

    default_pfas = "PFOA" if "PFOA" in pfas_names else pfas_names[0]
    default_soil = staring_soils[0]

    selection_form = (
        mo.md(
            """
            **PFAS** (select one or several)  
            {pfas}

            **Soil**  
            {soil}

            **Aaw method**  
            {aaw_method}

            **Grid depth** (total length of the soil column)  
            {domain_length} {domain_length_unit}

            **Target depth** (must be within the grid depth)  
            {depth} {depth_unit}

            **Total simulation time**  
            {sim_time} {time_unit}

            **Average infiltration rate**  
            {infiltration} {infiltration_unit}

            **Pulse concentration** (inlet, from the start of the model time)  
            {pulse_conc} {pulse_conc_unit}

            **Pulse duration** (concentration is 0 afterwards)  
            {pulse_duration} {pulse_duration_unit}
            """
        )
        .batch(
            pfas=mo.ui.multiselect(
                options=pfas_names, value=[default_pfas], label="",
                full_width=True,
            ),
            soil=mo.ui.dropdown(
                options=staring_soils, value=default_soil, label="",
                searchable=True, full_width=True,
            ),
            aaw_method=mo.ui.dropdown(
                options=aaw_method_options,
                value="Thermodynamic (SWC sorption)",
                label="",
                full_width=True,
            ),
            domain_length=mo.ui.number(start=0.001, value=100, step=1, label=""),
            domain_length_unit=mo.ui.dropdown(
                options=depth_unit_options, value="cm", label="",
            ),
            depth=mo.ui.number(start=0.001, value=50, step=1, label=""),
            depth_unit=mo.ui.dropdown(
                options=depth_unit_options, value="cm", label="",
            ),
            sim_time=mo.ui.number(start=1, value=250, step=1, label=""),
            time_unit=mo.ui.dropdown(
                options=time_unit_options, value="years", label="",
            ),
            # Default of 30 cm/year is ~ 9.51e-7 cm/s.
            infiltration=mo.ui.number(start=0.000001, value=30, step=0.1, label=""),
            infiltration_unit=mo.ui.dropdown(
                options=infiltration_unit_options, value="cm/year", label="",
            ),
            # Defaults (1 nmol/L for 25 years) reproduce the previous fixed pulse.
            pulse_conc=mo.ui.number(start=0.000001, value=1, step=0.1, label=""),
            pulse_conc_unit=mo.ui.dropdown(
                options=pulse_concentration_unit_options, value="nmol/L", label="",
            ),
            pulse_duration=mo.ui.number(start=0.001, value=25, step=1, label=""),
            pulse_duration_unit=mo.ui.dropdown(
                options=pulse_duration_unit_options, value="years", label="",
            ),
        )
        .form(submit_button_label="Run simulation")
    )

    # All inputs live in the sidebar so they stay visible next to the results.
    mo.sidebar([mo.md("## Inputs"), selection_form])
    return (selection_form,)


@app.cell(hide_code=True)
def _():
    def format_quantity(value, max_chars=60):
        """Compact string for a pint quantity / array / number."""
        try:
            text = f"{value:~.4g}"
        except Exception:
            text = str(value)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def run_selected_simulation(
        pfas_name, soil_name, aaw_method, pfas_db, soil_db,
        domain_length, target_depth, total_time, infiltration_rate,
        pulse_concentration, pulse_duration,
    ):
        """domain_length, target_depth, total_time, infiltration_rate,
        pulse_concentration and pulse_duration are (value, unit_str) tuples."""
        from pfas import ureg
        from pfas.component import (
            BoundaryPreprocessor, EquilibriumSolver, GridGenerator,
            Le2021_langmuir, LinearSPsorption, Retardation, Szyszkowski,
            WaterPreprocessor,
        )
        from pfas.component.awi import (
            D50AWI, GSSAAWI, NonlinearD50AWI, SWCsorption,
        )
        from pfas.model import Model

        spatial_resolution = 0.5 * ureg.centimeter
        time_resolution = 1 * ureg.month

        # Build user-chosen quantities (with units).
        domain_length_q = ureg.Quantity(domain_length[0], domain_length[1])
        target_depth_q = ureg.Quantity(target_depth[0], target_depth[1])
        time_total_q = ureg.Quantity(total_time[0], total_time[1])
        infiltration_q = ureg.Quantity(infiltration_rate[0], infiltration_rate[1])
        pulse_concentration_q = ureg.Quantity(pulse_concentration[0], pulse_concentration[1])
        pulse_duration_q = ureg.Quantity(pulse_duration[0], pulse_duration[1])

        if domain_length_q < spatial_resolution:
            raise ValueError(
                f"Grid depth must be at least {spatial_resolution:~P} "
                "(the spatial resolution)."
            )
        if target_depth_q > domain_length_q:
            raise ValueError(
                f"Target depth ({target_depth_q.to('cm'):~.4g}) exceeds the "
                f"grid depth ({domain_length_q.to('cm'):~.4g})."
            )
        if time_total_q < time_resolution:
            raise ValueError(
                f"Total simulation time must be at least {time_resolution:~P}."
            )

        selected_pfas = pfas_db[pfas_name]
        selected_soil = soil_db[soil_name]
        bulk_density = selected_soil["rho_b"]
        porosity = selected_soil["porosity"]
        theta_r = selected_soil["theta_r"]
        k_sat = selected_soil["K_sat"]
        vg_alpha = selected_soil["van_genuchten"]["alpha"]
        vg_n = selected_soil["van_genuchten"]["n"]
        vg_l = selected_soil["van_genuchten"]["l"]
        dispersivity = 4.5 * ureg.centimeter
        f_oc = selected_soil["f_oc"].to_base_units()
        f_silt_clay = selected_soil["f_silt"].to_base_units() + selected_soil["f_clay"].to_base_units()
        d50 = selected_soil["d50"]
        # Molar input -> multiply by this PFAS's molar mass; mass input -> as-is.
        if pulse_concentration_q.check("[substance]/[length]**3"):
            inlet_concentration = (pulse_concentration_q * selected_pfas["M"]).to("mg/L")
        else:
            inlet_concentration = pulse_concentration_q.to("mg/L")

        simulation_model = Model()
        simulation_model.compute(
            GridGenerator,
            domain_length=domain_length_q,
            spatial_resolution=spatial_resolution,
            time_resolution=time_resolution,
            time_total=time_total_q,
        )
        simulation_model.compute(
            WaterPreprocessor,
            average_infiltration_rate=infiltration_q.to("cm/s"),
            hydraulic_conductivity=k_sat,
            porosity=porosity,
            dispersivity=dispersivity,
            van_genuchten_n=vg_n,
            van_genuchten_l=vg_l,
            residual_water_content=theta_r,
        )
        simulation_model.compute(
            BoundaryPreprocessor,
            C_list=[inlet_concentration, ureg("0 mg/L")],
            T_list=[0.0 * ureg.years, pulse_duration_q],
        )

        # Fixed Kd configuration for this first GUI version.
        sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": "linear",
            "linear": {
                "Kd_method": "fabregat_palau",
                "n_CFx": selected_pfas["structural_properties"]["n_CFx"],
                "f_oc": f_oc,
                "f_silt_clay": f_silt_clay,
            },
        }
        simulation_model.compute(LinearSPsorption, sorption_solid=sorption_solid)

        # Fixed Kaw selection: Szyszkowski when parameters exist; otherwise Le2021.
        szyszkowski_params = selected_pfas["Szyszkowski_params"]
        szyszkowski_a = szyszkowski_params["a"]
        szyszkowski_b = szyszkowski_params["b"]
        if szyszkowski_a is not None and szyszkowski_b is not None:
            simulation_model.compute(
                Szyszkowski,
                sigma0=ureg("71 dyn/cm"),
                a=szyszkowski_a,
                b=szyszkowski_b,
                chi=1,
                T=293.15 * ureg.kelvin,
                Cw=ureg("1e-12 mol/L"),
            )
        else:
            simulation_model.compute(
                Le2021_langmuir,
                structural_properties=selected_pfas["structural_properties"],
                Cw=ureg("1e-12 mol/L"),
            )
            simulation_model.input_data["sigma0"] = ureg("71 dyn/cm")

        aaw_methods = {
            "thermo": (
                SWCsorption,
                {"scaling_factor_awi": 4.15, "van_genuchten_alpha": vg_alpha},
            ),
            "func_GSSA": (
                GSSAAWI,
                {"soil": {"porosity": porosity, "d50": d50}},
            ),
            "func_d50": (
                D50AWI,
                {"soil": {"porosity": porosity, "d50": d50}},
            ),
            "func_nonlin_d50": (
                NonlinearD50AWI,
                {"soil": {"porosity": porosity, "d50": d50}},
            ),
        }
        if aaw_method not in aaw_methods:
            raise ValueError(f"Unknown Aaw method: {aaw_method}")

        aaw_class, aaw_kwargs = aaw_methods[aaw_method]
        simulation_model.compute(aaw_class, **aaw_kwargs)
        aaw_value = simulation_model.aaw
        simulation_model.compute(Retardation, bulk_density=bulk_density)
        retardation_value = simulation_model.adsorption
        simulation_model.compute(EquilibriumSolver)

        return {
            "model": simulation_model,
            "pfas_name": pfas_name,
            "soil_name": soil_name,
            "aaw_method": aaw_method,
            "domain_length": domain_length_q,
            "target_depth": target_depth_q,
            "total_time": time_total_q,
            "infiltration_rate": infiltration_q,
            "pulse_concentration": pulse_concentration_q,
            "pulse_duration": pulse_duration_q,
            "inlet_concentration": inlet_concentration,
            "aaw": aaw_value,
            "kaw": simulation_model.Kaw,
            "kd": simulation_model.Kd,
            "retardation": retardation_value,
            "concentration": simulation_model.C1,
            "ureg": ureg,
        }

    return format_quantity, run_selected_simulation


@app.cell(hide_code=True)
def _(mo, pfas_db, run_selected_simulation, selection_form, soil_db):
    # Runs one simulation per selected PFAS (a single selection works too).
    # One failing PFAS does not stop the others.
    simulation_results = {}
    simulation_errors = {}

    _v = selection_form.value
    if _v is None:
        run_status = mo.md("Choose the inputs in the sidebar and click **Run simulation**.")
    elif not _v["pfas"]:
        run_status = mo.md("**Invalid input:** select at least one PFAS.").callout(kind="danger")
    else:
        for _name in mo.status.progress_bar(
            list(_v["pfas"]), title="Running simulations", remove_on_exit=True
        ):
            try:
                simulation_results[_name] = run_selected_simulation(
                    pfas_name=_name,
                    soil_name=_v["soil"],
                    aaw_method=_v["aaw_method"],
                    pfas_db=pfas_db,
                    soil_db=soil_db,
                    domain_length=(_v["domain_length"], _v["domain_length_unit"]),
                    target_depth=(_v["depth"], _v["depth_unit"]),
                    total_time=(_v["sim_time"], _v["time_unit"]),
                    infiltration_rate=(_v["infiltration"], _v["infiltration_unit"]),
                    pulse_concentration=(_v["pulse_conc"], _v["pulse_conc_unit"]),
                    pulse_duration=(_v["pulse_duration"], _v["pulse_duration_unit"]),
                )
            except Exception as _err:
                simulation_errors[_name] = f"{type(_err).__name__}: {_err}"

        _error_lines = "\n".join(f"- **{k}:** {m}" for k, m in simulation_errors.items())
        if simulation_errors and not simulation_results:
            run_status = mo.md(f"**All simulations failed:**\n\n{_error_lines}").callout(kind="danger")
        elif simulation_errors:
            run_status = mo.md(
                f"**{len(simulation_results)} of {len(_v['pfas'])} simulations completed.** "
                f"These failed:\n\n{_error_lines}"
            ).callout(kind="warn")
        else:
            run_status = mo.md(
                f"{len(simulation_results)} simulation(s) complete."
            ).callout(kind="success")

    # Time unit chosen by the user; used for the plot's x-axis.
    plot_time_unit = _v["time_unit"] if _v is not None else "year"
    return plot_time_unit, run_status, simulation_results


@app.cell(hide_code=True)
def _(mo):
    # Defined separately so changing it redraws the plot without re-running simulations.
    plot_mode = mo.ui.dropdown(
        options={
            "Concentration (mg/L)": "absolute",
            "Relative (C/C0)": "relative",
        },
        value="Concentration (mg/L)",
        label="Y-axis",
    )
    return (plot_mode,)


@app.cell(hide_code=True)
def _(format_quantity, mo, simulation_results):
    if not simulation_results:
        overview_tab = mo.md("Results will appear here after the first run.")
    else:
        _first = next(iter(simulation_results.values()))
        # Inputs shared by all PFAS in this run.
        _stats = mo.hstack(
            [
                mo.stat(label="Soil", value=_first["soil_name"]),
                mo.stat(label="Aaw method", value=_first["aaw_method"]),
                mo.stat(label="Grid depth", value=format_quantity(_first["domain_length"])),
                mo.stat(label="Target depth", value=format_quantity(_first["target_depth"])),
                mo.stat(label="Total time", value=format_quantity(_first["total_time"])),
                mo.stat(label="Infiltration", value=format_quantity(_first["infiltration_rate"])),
                mo.stat(label="Pulse concentration", value=format_quantity(_first["pulse_concentration"])),
                mo.stat(label="Pulse duration", value=format_quantity(_first["pulse_duration"])),
            ],
            justify="start",
            wrap=True,
        )
        # One row per PFAS.
        _rows = [
            {
                "PFAS": r["pfas_name"],
                "Inlet concentration": format_quantity(r["inlet_concentration"]),
                "Aaw": format_quantity(r["aaw"]),
                "Kaw": format_quantity(r["kaw"]),
                "Kd": format_quantity(r["kd"]),
            }
            for r in simulation_results.values()
        ]
        overview_tab = mo.vstack(
            [
                mo.md("### Shared inputs"),
                _stats,
                mo.md("### Results per PFAS"),
                mo.ui.table(_rows, selection=None, pagination=False),
            ]
        )
    return (overview_tab,)


@app.cell(hide_code=True)
def _(mo, plot_mode, plot_time_unit, plt, simulation_results):
    if not simulation_results:
        plot_tab = mo.md("The breakthrough plot will appear after the simulation runs.")
    else:
        _fig, _ax = plt.subplots(figsize=(9, 5))
        _relative = plot_mode.value == "relative"
        _depth_label = None
        for _res in simulation_results.values():
            _model = _res["model"]
            _ureg = _res["ureg"]
            _target_cm = _res["target_depth"].to(_ureg.centimeter).magnitude
            _idx = min(
                range(len(_model.grid.depth)),
                key=lambda i: abs(
                    _model.grid.depth[i].to(_ureg.centimeter).magnitude - _target_cm
                ),
            )
            _depth_label = _model.grid.depth[_idx].to(_ureg.centimeter)
            _t = _model.grid.time.to(plot_time_unit).magnitude
            _c = _res["concentration"][_idx, :]
            if hasattr(_c, "to"):
                _c = _c.to(_ureg.mg / _ureg.L).magnitude
            if _relative:
                _c = _c / _res["inlet_concentration"].to(_ureg.mg / _ureg.L).magnitude
            _ax.plot(_t, _c, linewidth=2, label=_res["pfas_name"])

        _ax.set_xlabel(f"Time ({plot_time_unit}s)")
        _ax.set_ylabel(
            "Relative concentration C/C0 (-)" if _relative
            else "Total PFAS concentration (mg/L)"
        )
        _ax.set_title(f"Breakthrough at {_depth_label:~P}")
        _ax.grid(True, alpha=0.3)
        if len(simulation_results) > 1:
            _ax.legend(title="PFAS")
        _fig.tight_layout()
        plot_tab = mo.vstack([plot_mode, mo.mpl.interactive(_fig)])
    return (plot_tab,)


@app.cell(hide_code=True)
def _(format_quantity, mo, simulation_results):
    if not simulation_results:
        details_tab = mo.md("Details will appear here after the first run.")
    else:
        _items = {
            _name: mo.md(
                "\n".join(
                    [
                        f"- **PFAS:** {_r['pfas_name']}",
                        f"- **Inlet concentration:** {format_quantity(_r['inlet_concentration'])}",
                        f"- **Pulse duration:** {format_quantity(_r['pulse_duration'])}",
                        f"- **Aaw:** {_r['aaw']}",
                        f"- **Kaw:** {_r['kaw']}",
                        f"- **Kd:** {_r['kd']}",
                    ]
                )
            )
            for _name, _r in simulation_results.items()
        }
        details_tab = mo.accordion(_items, multiple=True)
    return (details_tab,)


@app.cell(hide_code=True)
def _(details_tab, mo, overview_tab, plot_tab, run_status, simulation_results):
    # Final layout: status banner on top, results in tabs underneath.
    if simulation_results:
        _body = mo.ui.tabs(
            {
                "Overview": overview_tab,
                "Breakthrough curves": plot_tab,
                "Details per PFAS": details_tab,
            }
        )
        main_layout = mo.vstack([run_status, _body])
    else:
        main_layout = run_status
    main_layout
    return


if __name__ == "__main__":
    app.run()