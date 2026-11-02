import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    simulation_intro = mo.md(
        """
        # PFAS soil simulation

        Configure and run equilibrium PFAS transport simulations through a
        Staring soil profile. Select PFAS, soil, solid-phase sorption (Kd) method,
        air-water interfacial adsorption (Aaw) method, grid depth, target depth and
        transport conditions, and an inlet pulse, then inspect breakthrough curves
        and derived parameters.
        """
    )
    return (simulation_intro,)


@app.cell(hide_code=True)
def _():
    from matplotlib import pyplot as plt
    from pfas.data_loader import load_dataset


    pfas_db = load_dataset("PFASs")
    soil_db = load_dataset("soils")
    preferred_pfas_order = [
        "TFA", "PFBA", "HFPO-DA", "PFPeA", "PFBS", "PFHxA",
        "PFHpA", "PFHxS", "PFOA", "PFNA", "PFOS", "PFDA",
    ]
    pfas_names = [name for name in preferred_pfas_order if name in pfas_db]
    pfas_names += sorted(name for name in pfas_db if name not in pfas_names)
    staring_soils = sorted(name for name in soil_db if name.startswith("Staring-O"))

    if not pfas_names:
        raise ValueError("No PFAS entries were found in the PFASs dataset.")
    if not staring_soils:
        raise ValueError("No soils starting with 'Staring-O' were found.")
    return pfas_db, pfas_names, plt, soil_db, staring_soils


@app.cell(hide_code=True)
def _():
    # ── Solid-phase sorption (Kd) method registry ─────────────────────────────
    # One entry per method. The dropdown and the `sorption_solid` dict passed
    # to the model are both generated from this data; users enter no Kd values.
    #
    #   label      shown in the dropdown
    #   component  class name in `pfas.component`
    #   isotherm   key of the sorption_solid dict
    #   params     what goes into sorption_solid[isotherm]; a string "$name"
    #              is looked up in the context built from PFAS and soil data
    #              (see `build_kd_context` in the run cell)
    KD_METHODS = {
        "fabregat_palau": {
            "label": "Fabregat-Palau (2021): from n_CFx, f_oc, f_silt_clay",
            "component": "LinearSPsorption",
            "isotherm": "linear",
            "params": {
                "Kd_method": "fabregat_palau",
                "n_CFx": "$n_CFx",
                "f_oc": "$f_oc",
                "f_silt_clay": "$f_silt_clay",
            },
        },
        "linear_koc_ksc": {
            "label": "Linear Kd: from tabulated K_oc and K_sc",
            "component": "LinearSPsorption",
            "isotherm": "linear",
            "params": {"Kd_method": "direct_input", "Kd": "$Kd_tabulated"},
        },
    }
    default_kd_method = "fabregat_palau"
    kd_method_options = {spec["label"]: key for key, spec in KD_METHODS.items()}
    return KD_METHODS, default_kd_method, kd_method_options


@app.cell(hide_code=True)
def _(default_kd_method, kd_method_options, mo, pfas_names, staring_soils):
    aaw_method_options = {
        "Thermodynamic (SWC sorption)": "thermo",
        "GSSA-based": "func_GSSA",
        "d50-based": "func_d50",
        "d50-based, nonlinear saturation": "func_nonlin_d50",
    }
    depth_unit_options = {"cm": "centimeter", "mm": "millimeter", "m": "meter"}
    time_unit_options = {
        "years": "year", "months": "month", "days": "day", "seconds": "second",
    }
    infiltration_unit_options = {
        "cm/s": "cm/s", "mm/day": "mm/day", "cm/day": "cm/day",
        "mm/year": "mm/year", "cm/year": "cm/year", "m/year": "m/year",
    }
    pulse_concentration_unit_options = {
        "nmol/L": "nanomole/L", "µmol/L": "micromole/L", "mol/L": "mole/L",
        "ng/L": "nanogram/L", "µg/L": "microgram/L", "mg/L": "milligram/L",
    }
    default_pfas = "PFOA" if "PFOA" in pfas_names else pfas_names[0]
    default_kd_label = next(l for l, k in kd_method_options.items() if k == default_kd_method)

    selection_form = (
        mo.md(
            """
            **PFAS** (select one or several)  
            {pfas}

            **Soil**  
            {soil}

            **Kd method** (solid-phase sorption)  
            {kd_method}

            **Aaw method**  
            {aaw_method}

            **Grid depth** (total soil-column length)  
            {domain_length} {domain_length_unit}

            **Target depth** (within grid depth)  
            {depth} {depth_unit}

            **Total simulation time**  
            {sim_time} {time_unit}

            **Average infiltration rate**  
            {infiltration} {infiltration_unit}

            **Pulse concentration** (inlet concentration)  
            {pulse_conc} {pulse_conc_unit}

            **Pulse duration**  
            {pulse_duration} {pulse_duration_unit}
            """
        )
        .batch(
            pfas=mo.ui.multiselect(pfas_names, value=[default_pfas], label="", full_width=True),
            soil=mo.ui.dropdown(staring_soils, value=staring_soils[0], label="", searchable=True, full_width=True),
            kd_method=mo.ui.dropdown(kd_method_options, value=default_kd_label, label="", full_width=True),
            aaw_method=mo.ui.dropdown(aaw_method_options, value="Thermodynamic (SWC sorption)", label="", full_width=True),
            domain_length=mo.ui.number(start=0.001, value=100, step=1, label=""),
            domain_length_unit=mo.ui.dropdown(depth_unit_options, value="cm", label=""),
            depth=mo.ui.number(start=0.001, value=50, step=1, label=""),
            depth_unit=mo.ui.dropdown(depth_unit_options, value="cm", label=""),
            sim_time=mo.ui.number(start=1, value=250, step=1, label=""),
            time_unit=mo.ui.dropdown(time_unit_options, value="years", label=""),
            infiltration=mo.ui.number(start=0.000001, value=30, step=0.1, label=""),
            infiltration_unit=mo.ui.dropdown(infiltration_unit_options, value="cm/year", label=""),
            pulse_conc=mo.ui.number(start=0.000001, value=1, step=0.1, label=""),
            pulse_conc_unit=mo.ui.dropdown(pulse_concentration_unit_options, value="nmol/L", label=""),
            pulse_duration=mo.ui.number(start=0.001, value=25, step=1, label=""),
            pulse_duration_unit=mo.ui.dropdown(time_unit_options, value="years", label=""),
        )
        .form(submit_button_label="Run simulation")
    )
    return (selection_form,)


@app.cell(hide_code=True)
def _(mo):
    # Global navigation only: model inputs are displayed on the Simulation route.
    mo.sidebar(
        [
            mo.md("# PFAS soil model"),
            mo.nav_menu(
                {
                    "#/": f"{mo.icon('lucide:house')} Start",
                    "#/simulation": f"{mo.icon('lucide:flask-conical')} Simulation",
                    "#/slider-simulation": f"{mo.icon('lucide:sliders-horizontal')} Slider simulation",
                },
                orientation="vertical",
            ),
        ],
        width="16rem",
    )
    return


@app.cell(hide_code=True)
def _(KD_METHODS):
    def format_quantity(value, max_chars=60):
        try:
            text = f"{value:~.4g}"
        except Exception:
            text = str(value)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def build_kd_context(kd_method, pfas, soil, f_oc=None, f_silt_clay=None):
        """Named values that the `$name` references in KD_METHODS[...]["params"] resolve to.

        Everything comes from the datasets, unless `f_oc` / `f_silt_clay` (dimensionless
        quantities) are given, which then replace the soil's own values (slider page).
        """
        if f_oc is None:
            f_oc = soil["f_oc"]
        if f_silt_clay is None:
            f_silt_clay = soil["f_silt"].to_base_units() + soil["f_clay"].to_base_units()
        f_oc = f_oc.to_base_units()
        f_silt_clay = f_silt_clay.to_base_units()
        context = {
            "n_CFx": pfas["structural_properties"]["n_CFx"],
            "f_oc": f_oc,
            "f_silt_clay": f_silt_clay.magnitude,
        }
        # Only needed by the tabulated linear method (K_oc / K_sc may be absent for other PFAS).
        if kd_method == "linear_koc_ksc":
            context["Kd_tabulated"] = pfas["K_oc"] * f_oc + pfas["K_sc"] * f_silt_clay.magnitude
        return context

    def resolve(value, context):
        if isinstance(value, str) and value.startswith("$"):
            return context[value[1:]]
        return value

    def run_selected_simulation(
        pfas_name, soil_name, aaw_method, pfas_db, soil_db,
        domain_length, target_depth, total_time, infiltration_rate,
        pulse_concentration, pulse_duration, kd_method,
        f_oc=None, f_silt_clay=None,
    ):
        import pfas.component as pfas_components
        from pfas import ureg
        from pfas.component import (
            BoundaryPreprocessor, EquilibriumSolver, GridGenerator,
            Le2021_langmuir, Retardation, Szyszkowski, WaterPreprocessor,
        )
        from pfas.component.awi import D50AWI, GSSAAWI, NonlinearD50AWI, SWCsorption
        from pfas.model import Model

        spatial_resolution = 0.5 * ureg.centimeter
        time_resolution = 1 * ureg.month
        domain_length_q = ureg.Quantity(*domain_length)
        target_depth_q = ureg.Quantity(*target_depth)
        total_time_q = ureg.Quantity(*total_time)
        infiltration_q = ureg.Quantity(*infiltration_rate)
        pulse_concentration_q = ureg.Quantity(*pulse_concentration)
        pulse_duration_q = ureg.Quantity(*pulse_duration)

        if domain_length_q < spatial_resolution:
            raise ValueError(f"Grid depth must be at least {spatial_resolution:~P}.")
        if target_depth_q > domain_length_q:
            raise ValueError("Target depth cannot exceed grid depth.")
        if total_time_q < time_resolution:
            raise ValueError(f"Total simulation time must be at least {time_resolution:~P}.")

        pfas = pfas_db[pfas_name]
        soil = soil_db[soil_name]
        if pulse_concentration_q.check("[substance]/[length]**3"):
            inlet_concentration = (pulse_concentration_q * pfas["M"]).to("mg/L")
        else:
            inlet_concentration = pulse_concentration_q.to("mg/L")

        model = Model()
        model.compute(
            GridGenerator,
            domain_length=domain_length_q,
            spatial_resolution=spatial_resolution,
            time_resolution=time_resolution,
            time_total=total_time_q,
        )
        model.compute(
            WaterPreprocessor,
            average_infiltration_rate=infiltration_q.to("cm/s"),
            hydraulic_conductivity=soil["K_sat"],
            porosity=soil["porosity"],
            dispersivity=4.5 * ureg.centimeter,
            van_genuchten_n=soil["van_genuchten"]["n"],
            van_genuchten_l=soil["van_genuchten"]["l"],
            residual_water_content=soil["theta_r"],
        )
        model.compute(
            BoundaryPreprocessor,
            C_list=[inlet_concentration, ureg("0 mg/L")],
            T_list=[0.0 * ureg.year, pulse_duration_q],
        )

        # ── Solid-phase sorption, generated from the KD_METHODS registry ──────
        try:
            kd_spec = KD_METHODS[kd_method]
        except KeyError as error:
            raise ValueError(f"Unknown Kd method: {kd_method}") from error
        kd_context = build_kd_context(kd_method, pfas, soil, f_oc, f_silt_clay)
        sorption_solid = {
            "kinetic_sorption": False,
            "sorption_isotherm": kd_spec["isotherm"],
            "kinetic": {"frac_int": 0.3, "rate_const": 0.01 / ureg.s},
            kd_spec["isotherm"]: {
                key: resolve(value, kd_context) for key, value in kd_spec["params"].items()
            },
        }
        model.compute(getattr(pfas_components, kd_spec["component"]), sorption_solid=sorption_solid)

        parameters = pfas["Szyszkowski_params"]
        if parameters["a"] is not None and parameters["b"] is not None:
            model.compute(
                Szyszkowski, sigma0=ureg("71 dyn/cm"), a=parameters["a"], b=parameters["b"],
                chi=1, T=293.15 * ureg.kelvin, Cw=ureg("1e-12 mol/L"),
            )
        else:
            model.compute(Le2021_langmuir, structural_properties=pfas["structural_properties"], Cw=ureg("1e-12 mol/L"))
            model.input_data["sigma0"] = ureg("71 dyn/cm")

        aaw_models = {
            "thermo": (SWCsorption, {"scaling_factor_awi": 4.15, "van_genuchten_alpha": soil["van_genuchten"]["alpha"]}),
            "func_GSSA": (GSSAAWI, {"soil": {"porosity": soil["porosity"], "d50": soil["d50"]}}),
            "func_d50": (D50AWI, {"soil": {"porosity": soil["porosity"], "d50": soil["d50"]}}),
            "func_nonlin_d50": (NonlinearD50AWI, {"soil": {"porosity": soil["porosity"], "d50": soil["d50"]}}),
        }
        try:
            aaw_class, aaw_kwargs = aaw_models[aaw_method]
        except KeyError as error:
            raise ValueError(f"Unknown Aaw method: {aaw_method}") from error
        model.compute(aaw_class, **aaw_kwargs)
        aaw = model.aaw
        model.compute(Retardation, bulk_density=soil["rho_b"])
        retardation = model.adsorption
        model.compute(EquilibriumSolver)

        return {
            "model": model, "ureg": ureg, "pfas_name": pfas_name, "soil_name": soil_name,
            "aaw_method": aaw_method, "kd_method": kd_spec["label"],
            "domain_length": domain_length_q,
            "target_depth": target_depth_q, "total_time": total_time_q,
            "infiltration_rate": infiltration_q, "pulse_concentration": pulse_concentration_q,
            "pulse_duration": pulse_duration_q, "inlet_concentration": inlet_concentration,
            "aaw": aaw, "kaw": model.Kaw, "kd": model.Kd,
            "retardation": retardation, "concentration": model.C1,
        }

    return format_quantity, run_selected_simulation


@app.cell(hide_code=True)
def _(mo, pfas_db, run_selected_simulation, selection_form, soil_db):
    values = selection_form.value
    simulation_results = {}
    errors = {}
    if values is None:
        run_status = mo.md("Set the inputs and click **Run simulation**.")
    elif not values["pfas"]:
        run_status = mo.md("**Invalid input:** select at least one PFAS.").callout(kind="danger")
    else:
        for name in mo.status.progress_bar(list(values["pfas"]), title="Running simulations", remove_on_exit=True):
            try:
                simulation_results[name] = run_selected_simulation(
                    name, values["soil"], values["aaw_method"], pfas_db, soil_db,
                    (values["domain_length"], values["domain_length_unit"]),
                    (values["depth"], values["depth_unit"]),
                    (values["sim_time"], values["time_unit"]),
                    (values["infiltration"], values["infiltration_unit"]),
                    (values["pulse_conc"], values["pulse_conc_unit"]),
                    (values["pulse_duration"], values["pulse_duration_unit"]),
                    values["kd_method"],
                )
            except Exception as error:
                errors[name] = f"{type(error).__name__}: {error}"
        error_text = "\n".join(f"- **{name}:** {message}" for name, message in errors.items())
        if errors and not simulation_results:
            run_status = mo.md(f"**All simulations failed:**\n\n{error_text}").callout(kind="danger")
        elif errors:
            run_status = mo.md(f"**{len(simulation_results)} simulations completed.**\n\n{error_text}").callout(kind="warn")
        else:
            run_status = mo.md(f"{len(simulation_results)} simulation(s) complete.").callout(kind="success")
    plot_time_unit = values["time_unit"] if values is not None else "year"
    return plot_time_unit, run_status, simulation_results


@app.cell(hide_code=True)
def _(mo):
    plot_mode = mo.ui.dropdown(
        {"Concentration (mg/L)": "absolute", "Relative (C/C0)": "relative"},
        value="Concentration (mg/L)", label="Y-axis",
    )
    return (plot_mode,)


@app.cell(hide_code=True)
def _(format_quantity, mo, simulation_results):
    if not simulation_results:
        overview_tab = mo.md("Results will appear here after the first run.")
    else:
        first = next(iter(simulation_results.values()))
        stats = mo.hstack([
            mo.stat("Soil", first["soil_name"]), mo.stat("Aaw method", first["aaw_method"]),
            mo.stat("Kd method", first["kd_method"].split(":")[0]),
            mo.stat("Grid depth", format_quantity(first["domain_length"])),
            mo.stat("Target depth", format_quantity(first["target_depth"])),
            mo.stat("Total time", format_quantity(first["total_time"])),
            mo.stat("Infiltration", format_quantity(first["infiltration_rate"])),
        ], justify="start", wrap=True)
        rows = [{
            "PFAS": result["pfas_name"],
            "Inlet concentration": format_quantity(result["inlet_concentration"]),
            "Aaw": format_quantity(result["aaw"]), "Kaw": format_quantity(result["kaw"]),
            "Kd": format_quantity(result["kd"]),
        } for result in simulation_results.values()]
        overview_tab = mo.vstack([mo.md("### Shared inputs"), stats, mo.md("### Results per PFAS"), mo.ui.table(rows, selection=None, pagination=False)])
    return (overview_tab,)


@app.cell(hide_code=True)
def _(mo, plot_mode, plot_time_unit, plt, simulation_results):
    import numpy as np
    if not simulation_results:
        plot_tab = mo.md("The breakthrough plot will appear after the simulation runs.")
    else:
        cmap = plt.get_cmap("viridis")  # or "plasma", "tab10", "coolwarm", ...
        colors = cmap(np.linspace(0.0, 0.9, max(len(simulation_results), 2)))
        figure, axis = plt.subplots(figsize=(9, 5))
        relative = plot_mode.value == "relative"
        depth_label = None
        for result, color in zip(simulation_results.values(), colors):
            model, ureg = result["model"], result["ureg"]
            target_cm = result["target_depth"].to(ureg.centimeter).magnitude
            index = min(range(len(model.grid.depth)), key=lambda i: abs(model.grid.depth[i].to(ureg.centimeter).magnitude - target_cm))
            depth_label = model.grid.depth[index].to(ureg.centimeter)
            concentration = result["concentration"][index, :]
            if hasattr(concentration, "to"):
                concentration = concentration.to("mg/L").magnitude
            if relative:
                concentration = concentration / result["inlet_concentration"].to("mg/L").magnitude
            axis.plot(model.grid.time.to(plot_time_unit).magnitude, concentration, linewidth=2, label=result["pfas_name"], color = color)
        axis.set_xlabel(f"Time ({plot_time_unit})")
        axis.set_ylabel("Relative concentration C/C0 (-)" if relative else "Total PFAS concentration (mg/L)")
        axis.set_title(f"Breakthrough at {depth_label:~P}")
        axis.grid(True, alpha=0.3)
        if len(simulation_results) > 1:
            axis.legend(title="PFAS")
        figure.tight_layout()
        plot_tab = mo.vstack([plot_mode, mo.mpl.interactive(figure)])
    return (plot_tab,)


@app.cell(hide_code=True)
def _(format_quantity, mo, simulation_results):
    if not simulation_results:
        details_tab = mo.md("Details will appear here after the first run.")
    else:
        details_tab = mo.accordion({
            name: mo.md("\n".join([
                f"- **PFAS:** {result['pfas_name']}",
                f"- **Kd method:** {result['kd_method']}",
                f"- **Inlet concentration:** {format_quantity(result['inlet_concentration'])}",
                f"- **Pulse duration:** {format_quantity(result['pulse_duration'])}",
                f"- **Aaw:** {format_quantity(result['aaw'])}",
                f"- **Kaw:** {format_quantity(result['kaw'])}",
                f"- **Kd:** {format_quantity(result['kd'])}",
            ])) for name, result in simulation_results.items()
        }, multiple=True)
    return (details_tab,)


@app.cell(hide_code=True)
def _(details_tab, mo, overview_tab, plot_tab, run_status, selection_form, simulation_intro, simulation_results):
    if simulation_results:
        results = mo.ui.tabs({"Overview": overview_tab, "Breakthrough curves": plot_tab, "Details per PFAS": details_tab})
        body = mo.vstack([run_status, results])
    else:
        body = run_status
    inputs = mo.vstack([mo.md("## Inputs"), selection_form])
    simulation_page = mo.vstack([
        simulation_intro,
        mo.hstack([inputs, body], justify="start", align="start", gap=2, wrap=True),
    ], gap=1)
    return (simulation_page,)


# ══════════════════════════════════════════════════════════════════════════════
# Slider simulation page
# One PFAS + one soil, with live sliders for infiltration rate, f_oc and f_silt_clay.
# Fixed on this page: Fabregat-Palau Kd and thermodynamic Aaw.
# ══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(mo, pfas_names, staring_soils):
    # Fixed settings of the slider page, as (value, unit) pairs like the form values.
    slider_fixed = {
        "kd_method": "fabregat_palau",
        "aaw_method": "thermo",
        "domain_length": (100, "centimeter"),
        "target_depth": (50, "centimeter"),
        "total_time": (250, "year"),
        "pulse_concentration": (1, "nanomole/L"),
        "pulse_duration": (25, "year"),
        "default_infiltration": 30,  # cm/year, also the slider's starting value
    }

    slider_pfas = mo.ui.dropdown(
        pfas_names, value="PFOA" if "PFOA" in pfas_names else pfas_names[0],
        label="PFAS", searchable=True, full_width=True,
    )
    slider_soil = mo.ui.dropdown(
        staring_soils, value=staring_soils[0], label="Soil", searchable=True, full_width=True,
    )
    # debounce=True: the simulation reruns when a slider is released, not on every pixel of movement.
    slider_infiltration = mo.ui.slider(
        start=1, stop=200, step=1, value=slider_fixed["default_infiltration"],
        label="Average infiltration rate (cm/year)",
        debounce=True, show_value=True, full_width=True,
    )
    return slider_fixed, slider_infiltration, slider_pfas, slider_soil


@app.cell(hide_code=True)
def _(mo, slider_soil, soil_db):
    # Rebuilt whenever the soil changes, so the sliders start at that soil's own values.
    _soil = soil_db[slider_soil.value]
    _f_oc_pct = float(_soil["f_oc"].to_base_units().magnitude) * 100
    _f_sc_pct = float((_soil["f_silt"].to_base_units() + _soil["f_clay"].to_base_units()).magnitude) * 100

    _oc_start, _oc_stop = 0.01, max(10.0, 2 * _f_oc_pct)
    _sc_start, _sc_stop = 0.1, 100.0

    slider_f_oc = mo.ui.slider(
        start=_oc_start, stop=_oc_stop, step=0.01,
        value=min(max(round(_f_oc_pct, 2), _oc_start), _oc_stop),
        label=f"Organic carbon fraction f_oc (%), soil default {_f_oc_pct:.3g}",
        debounce=True, show_value=True, full_width=True,
    )
    slider_f_sc = mo.ui.slider(
        start=_sc_start, stop=_sc_stop, step=0.1,
        value=min(max(round(_f_sc_pct, 1), _sc_start), _sc_stop),
        label=f"Silt + clay fraction f_sc (%), soil default {_f_sc_pct:.3g}",
        debounce=True, show_value=True, full_width=True,
    )
    return slider_f_oc, slider_f_sc


@app.cell(hide_code=True)
def _(pfas_db, run_selected_simulation, slider_fixed, soil_db):
    def run_slider_case(pfas_name, soil_name, infiltration, f_oc_pct=None, f_sc_pct=None):
        """One simulation with the fixed slider-page settings; None keeps the soil's own value."""
        from pfas import ureg

        f_oc_q = None if f_oc_pct is None else ureg.Quantity(f_oc_pct / 100, "dimensionless")
        f_sc_q = None if f_sc_pct is None else ureg.Quantity(f_sc_pct / 100, "dimensionless")
        return run_selected_simulation(
            pfas_name, soil_name, slider_fixed["aaw_method"], pfas_db, soil_db,
            slider_fixed["domain_length"], slider_fixed["target_depth"], slider_fixed["total_time"],
            (infiltration, "cm/year"), slider_fixed["pulse_concentration"], slider_fixed["pulse_duration"],
            slider_fixed["kd_method"], f_oc=f_oc_q, f_silt_clay=f_sc_q,
        )

    def breakthrough_curve(result):
        """Relative concentration C/C0 over time (years) at the grid point closest to the target depth."""
        _model, _ureg = result["model"], result["ureg"]
        _target_cm = result["target_depth"].to(_ureg.centimeter).magnitude
        _depths = [d.to(_ureg.centimeter).magnitude for d in _model.grid.depth]
        _index = min(range(len(_depths)), key=lambda i: abs(_depths[i] - _target_cm))
        _concentration = result["concentration"][_index, :]
        if hasattr(_concentration, "to"):
            _concentration = _concentration.to("mg/L").magnitude
        _relative = _concentration / result["inlet_concentration"].to("mg/L").magnitude
        return _model.grid.time.to("year").magnitude, _relative, _depths[_index]

    return breakthrough_curve, run_slider_case


@app.cell(hide_code=True)
def _(run_slider_case, slider_fixed, slider_pfas, slider_soil):
    # Reference run with the soil's own f_oc / f_sc and the default infiltration rate.
    # Depends only on PFAS and soil, so moving a slider does not recompute it.
    try:
        slider_baseline = run_slider_case(
            slider_pfas.value, slider_soil.value, slider_fixed["default_infiltration"],
        )
    except Exception:
        slider_baseline = None
    return (slider_baseline,)


@app.cell(hide_code=True)
def _(
    breakthrough_curve, format_quantity, mo, run_slider_case, slider_baseline,
    slider_f_oc, slider_f_sc, slider_infiltration, slider_pfas, slider_soil,
):
    from matplotlib import colormaps as _colormaps
    from matplotlib.figure import Figure as _Figure

    try:
        _current = run_slider_case(
            slider_pfas.value, slider_soil.value, slider_infiltration.value,
            slider_f_oc.value, slider_f_sc.value,
        )
    except Exception as _error:
        slider_view = mo.md(
            f"**Simulation failed:** `{type(_error).__name__}: {_error}`"
        ).callout(kind="danger")
    else:
        _time, _relative, _depth_cm = breakthrough_curve(_current)

        _figure = _Figure(figsize=(9, 5))
        _axis = _figure.subplots()
        if slider_baseline is not None:
            _time_ref, _relative_ref, _ = breakthrough_curve(slider_baseline)
            _axis.plot(_time_ref, _relative_ref, color="gray", linestyle="--", linewidth=2, label="Soil defaults")
        _axis.plot(
            _time, _relative, linewidth=2.5, color=_colormaps["viridis"](0.3),
            label=(
                f"Current: {slider_infiltration.value:g} cm/year, "
                f"f_oc {slider_f_oc.value:g}%, f_sc {slider_f_sc.value:g}%"
            ),
        )
        _axis.set_xlabel("Time (year)")
        _axis.set_ylabel("Relative concentration C/C0 (-)")
        _axis.set_ylim(0, 1.05)
        _axis.set_title(f"{slider_pfas.value} breakthrough in {slider_soil.value} at {_depth_cm:.1f} cm")
        _axis.grid(True, alpha=0.3)
        _axis.legend(loc="upper right")
        _figure.tight_layout()

        _reference_kd = format_quantity(slider_baseline["kd"]) if slider_baseline is not None else "n/a"
        _stats = mo.hstack([
            mo.stat("Kd", format_quantity(_current["kd"]), caption=f"soil defaults: {_reference_kd}"),
            mo.stat("Aaw", format_quantity(_current["aaw"])),
            mo.stat("Kaw", format_quantity(_current["kaw"])),
        ], justify="start", wrap=True)
        slider_view = mo.vstack([_figure, _stats], gap=1)
    return (slider_view,)


@app.cell(hide_code=True)
def _(
    mo, slider_f_oc, slider_f_sc, slider_infiltration, slider_pfas, slider_soil, slider_view,
):
    _controls = mo.vstack([
        mo.md("## Inputs"),
        slider_pfas,
        slider_soil,
        slider_infiltration,
        slider_f_oc,
        slider_f_sc,
        mo.md(
            "*Sliders update the plot when released. Fixed on this page: Fabregat-Palau Kd, "
            "thermodynamic Aaw, 100 cm grid, results at 50 cm, a 1 nmol/L pulse of 25 years, "
            "250 years total.*"
        ),
    ], gap=0.75)
    slider_simulation_page = mo.vstack([
        mo.md(
            """
            # Slider simulation

            Pick a PFAS and soil, then move the sliders to see how the infiltration rate and the
            soil composition (f_oc, f_silt_clay) change the breakthrough curve. The dashed gray
            line is the soil's own composition at the default infiltration rate.
            """
        ),
        mo.hstack([_controls, slider_view], widths=[1, 2], align="start", gap=2),
    ], gap=1)
    return (slider_simulation_page,)


@app.cell(hide_code=True)
def _(mo, simulation_page, slider_simulation_page):
    start_page = mo.vstack([
        mo.md(
            """
            # PFAS soil model

            This application provides tools for simulating PFAS transport conceptually in a 1D domain. 
            The simulation interface allows users to select PFAS and soil types, configure transport parameters, and visualize the resulting breakthrough curves.
            Please note that this is a simplified model and may not capture all complexities of PFAS transport in real-world scenarios.
            More advanced features are present in the pfas Python package. 
            """
        ),
        mo.md("## Choose an application"),
        mo.nav_menu(
            {
                "#/simulation": f"{mo.icon('lucide:flask-conical')} Open simulation",
                "#/slider-simulation": f"{mo.icon('lucide:sliders-horizontal')} Open slider simulation",
            },
            orientation="horizontal",
        ),
    ], gap=1)
    mo.routes({
        "#/": start_page,
        "#/simulation": simulation_page,
        "#/slider-simulation": slider_simulation_page,
        mo.routes.CATCH_ALL: start_page,
    })
    return


if __name__ == "__main__":
    app.run()