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
        Staring soil profile. Select PFAS, soil, transport conditions, and an
        inlet pulse, then inspect breakthrough curves and derived parameters.
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
def _(mo, pfas_names, staring_soils):
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

    selection_form = (
        mo.md(
            """
            **PFAS** (select one or several)  
            {pfas}

            **Soil**  
            {soil}

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
def _():
    def format_quantity(value, max_chars=60):
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
        from pfas import ureg
        from pfas.component import (
            BoundaryPreprocessor, EquilibriumSolver, GridGenerator,
            Le2021_langmuir, LinearSPsorption, Retardation, Szyszkowski,
            WaterPreprocessor,
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
        model.compute(
            LinearSPsorption,
            sorption_solid={
                "kinetic_sorption": False,
                "sorption_isotherm": "linear",
                "linear": {
                    "Kd_method": "fabregat_palau",
                    "n_CFx": pfas["structural_properties"]["n_CFx"],
                    "f_oc": soil["f_oc"].to_base_units(),
                    "f_silt_clay": soil["f_silt"].to_base_units() + soil["f_clay"].to_base_units(),
                },
            },
        )

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
            "aaw_method": aaw_method, "domain_length": domain_length_q,
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
    if not simulation_results:
        plot_tab = mo.md("The breakthrough plot will appear after the simulation runs.")
    else:
        figure, axis = plt.subplots(figsize=(9, 5))
        relative = plot_mode.value == "relative"
        depth_label = None
        for result in simulation_results.values():
            model, ureg = result["model"], result["ureg"]
            target_cm = result["target_depth"].to(ureg.centimeter).magnitude
            index = min(range(len(model.grid.depth)), key=lambda i: abs(model.grid.depth[i].to(ureg.centimeter).magnitude - target_cm))
            depth_label = model.grid.depth[index].to(ureg.centimeter)
            concentration = result["concentration"][index, :]
            if hasattr(concentration, "to"):
                concentration = concentration.to("mg/L").magnitude
            if relative:
                concentration = concentration / result["inlet_concentration"].to("mg/L").magnitude
            axis.plot(model.grid.time.to(plot_time_unit).magnitude, concentration, linewidth=2, label=result["pfas_name"])
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


@app.cell(hide_code=True)
def _(mo, simulation_page):
    start_page = mo.vstack([
        mo.md(
            """
            # PFAS soil model

            This application provides tools for simulating PFAS transport in a
            one-dimensional soil column. Choose an application to begin.
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
    slider_simulation_page = mo.vstack([
        mo.md("# Slider simulation"),
        mo.md("This page is reserved for the future slider-based simulation interface.").callout(kind="info"),
    ])
    mo.routes({
        "#/": start_page,
        "#/simulation": simulation_page,
        "#/slider-simulation": slider_simulation_page,
        mo.routes.CATCH_ALL: start_page,
    })
    return


if __name__ == "__main__":
    app.run()
