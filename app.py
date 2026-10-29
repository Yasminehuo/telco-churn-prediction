
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

from churn_scoring import ChurnScorer

scorer = ChurnScorer("churn_model_bundle.pkl")

model = scorer.model
threshold = scorer.threshold
final_features = scorer.final_features
dummy_groups = scorer.dummy_groups
category_values = scorer.category_values
binary_features = scorer.binary_features
what_if_features = scorer.what_if_features
tenure_max = scorer.tenure_max

customers = pd.read_csv("demo_customer_data.csv")

FEATURE_LABELS = {
    "tenure": "Tenure",
    "InternetService": "Internet Service",
    "Contract": "Contract",
    "PaymentMethod": "Payment Method",
    "SeniorCitizen": "Senior Citizen",
    "MultipleLines": "Multiple Lines",
    "OnlineSecurity": "Online Security",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing"
}

feature_choices = {f: FEATURE_LABELS.get(f, f) for f in what_if_features}
default_customer = str(customers["customerID"].iloc[0])
default_feature = "Contract" if "Contract" in what_if_features else what_if_features[0]

COLOR_BG = "#F4F5F7"
COLOR_PANEL = "#FFFFFF"
COLOR_PANEL_ALT = "#F0F1F4"
COLOR_BORDER = "#E1E3E8"
COLOR_TEXT = "#1C2026"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_RISK = "#C1512F"
COLOR_PROTECTIVE = "#2A8A73"
COLOR_PRIMARY = "#3D5A99"

mpl.rcParams.update({
    "figure.facecolor": COLOR_PANEL,
    "axes.facecolor": COLOR_PANEL,
    "savefig.facecolor": COLOR_PANEL,
    "text.color": COLOR_TEXT,
    "axes.edgecolor": COLOR_BORDER,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT_MUTED,
    "ytick.color": COLOR_TEXT_MUTED,
    "grid.color": COLOR_BORDER,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "Arial"],
    "font.size": 10.5,
})


def score_customer(customer_id):
    matched = customers[customers["customerID"].astype(str) == str(customer_id)]

    if matched.empty:
        return None

    customer_x = matched[final_features].astype(float)
    scoring_result = scorer.score(customer_x)

    actual = "Churn" if matched["Actual_Churn"].iloc[0] == 1 else "No Churn"

    return {
        "customer_x": customer_x,
        "probability": scoring_result["probability"],
        "prediction": scoring_result["prediction"],
        "actual": actual,
        "shap_values": scoring_result["shap_values"],
        "risk_drivers": scoring_result["risk_drivers"],
        "protective_drivers": scoring_result["protective_drivers"]
    }


def apply_scenario(customer_x, feature_name, value):
    return scorer.apply_scenario(customer_x, feature_name, value)


def get_sweep_values(feature_name):
    if feature_name == "tenure":
        values = np.arange(0, tenure_max + 1, 3)
        if tenure_max not in values:
            values = np.append(values, tenure_max)
        return values.tolist()

    if feature_name in dummy_groups:
        states = {}
        for value in category_values[feature_name]:
            expected_col = f"{feature_name}_{value}"
            state = expected_col if expected_col in dummy_groups[feature_name] else "__all_zero__"
            states.setdefault(state, []).append(value)
        return list(states.values())

    if feature_name in binary_features:
        return [[0], [1]]

    raise ValueError(f"No sweep rule for {feature_name}")


def what_if_sweep(customer_x, feature_name):
    rows = []

    for values in get_sweep_values(feature_name):
        if feature_name == "tenure":
            scenario_value = values
            display_label = str(int(values))
        else:
            scenario_value = values[0]
            display_label = str(values[0]) if len(values) == 1 else "Other / reference"

        scenario = apply_scenario(customer_x, feature_name, scenario_value)
        probability = model.predict_proba(scenario[final_features].astype(float))[0, 1]

        rows.append({
            "Scenario Value": display_label,
            "Scenario Raw Value": scenario_value,
            "Predicted Churn Probability": probability
        })

    return pd.DataFrame(rows)


def get_current_feature_value(customer_x, feature_name):
    row = customer_x.iloc[0]

    if feature_name == "tenure":
        return int(row["tenure"])

    if feature_name in binary_features:
        return "Yes" if int(row[feature_name]) == 1 else "No"

    if feature_name in dummy_groups:
        for value in category_values[feature_name]:
            col = f"{feature_name}_{value}"
            if col in dummy_groups[feature_name] and int(row[col]) == 1:
                return value

        refs = [v for v in category_values[feature_name] if f"{feature_name}_{v}" not in dummy_groups[feature_name]]
        return refs[0] if len(refs) == 1 else "Other / reference"

    return str(row.get(feature_name, ""))


def get_scenario_options(feature_name):
    if feature_name == "tenure":
        return None

    if feature_name in binary_features:
        return {"0": "No", "1": "Yes"}

    options = {}
    for values in get_sweep_values(feature_name):
        raw = values[0]
        label = str(raw) if len(values) == 1 else "Other / reference"
        options[str(raw)] = label

    return options


app_ui = ui.page_fluid(
    ui.tags.link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
    ),

    ui.tags.style("""
        :root {
            --bg: #F4F5F7;
            --panel: #FFFFFF;
            --panel-alt: #F0F1F4;
            --border: #E1E3E8;
            --text: #1C2026;
            --text-muted: #6B7280;
            --risk: #C1512F;
            --protective: #2A8A73;
            --primary: #3D5A99;
        }

        html, body {
            background: var(--bg);
            color: var(--text);
            font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
        }

        .container-fluid { padding: 20px 28px 32px; max-width: 1400px; }
        .dashboard-header { margin-bottom: 20px; }

        .dashboard-title {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--text);
        }

        .dashboard-subtitle {
            margin: 4px 0 0;
            font-size: 13.5px;
            color: var(--text-muted);
        }

        .bslib-sidebar-layout > .sidebar,
        .sidebar {
            background: var(--panel-alt) !important;
            border-right: 1px solid var(--border) !important;
            padding: 18px 16px !important;
        }

        .sidebar-block { margin-bottom: 18px; }

        .sidebar-heading {
            font-weight: 600;
            font-size: 13px;
            color: var(--text);
            margin-bottom: 8px;
        }

        hr { border-color: var(--border); opacity: 1; margin: 14px 0; }
        .subtle { color: var(--text-muted); font-size: 12px; }

        label, .control-label, .shiny-input-container label {
            color: var(--text-muted) !important;
            font-size: 12.5px !important;
            font-weight: 500 !important;
            margin-bottom: 4px !important;
        }

        .form-control, .form-select,
        select.form-control, input.form-control {
            background: var(--panel-alt) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 6px !important;
        }

        .form-control:focus, .form-select:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(61, 90, 153, 0.2) !important;
        }

        .btn-primary {
            background: var(--primary) !important;
            border-color: var(--primary) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
        }

        .btn-primary:hover { filter: brightness(1.08); }

        .irs--shiny .irs-bar, .irs--shiny .irs-single, .irs--shiny .irs-handle {
            background: var(--primary) !important;
            border-color: var(--primary) !important;
        }

        .irs--shiny .irs-line { background: var(--border) !important; }
        .irs--shiny .irs-grid-text { color: var(--text-muted) !important; }

        .irs--shiny .irs-min, .irs--shiny .irs-max {
            color: var(--text-muted) !important;
            background: transparent !important;
        }

        .section-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text);
            margin: 0 0 12px 0;
        }

        .summary-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 3px solid var(--primary);
            border-radius: 8px;
            padding: 12px 16px;
            height: 78px;
        }

        .summary-label {
            font-size: 11.5px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .summary-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 22px;
            font-weight: 600;
            color: var(--text);
        }

        .panel-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
        }

        .equal-height-row > .row { align-items: stretch; }
        .equal-height-row .panel-card { height: 100%; }

        .driver-panel {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 28px;
        }

        .driver-section {
            width: 100%;
            overflow-x: auto;
        }

        .driver-table {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            font-size: 12px;
        }

        .driver-table col.col-feature { width: 54%; }
        .driver-table col.col-value { width: 21%; }
        .driver-table col.col-shap { width: 25%; }

        .driver-table th {
            font-weight: 600;
            font-size: 10.5px;
            letter-spacing: 0.02em;
            color: var(--text-muted);
            text-align: left;
            padding: 4px 4px 6px;
            border-bottom: 1px solid var(--border);
        }

        .driver-table td {
            padding: 6px 4px;
            border-bottom: 1px solid var(--border);
            color: var(--text);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            vertical-align: top;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .driver-table td:first-child {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 12px;
            white-space: normal;
            word-break: break-word;
        }

        .driver-table tr:last-child td { border-bottom: none; }

        .shap-pill {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 10px;
            font-size: 10.5px;
            white-space: nowrap;
        }

        .shap-pill.risk {
            background: rgba(221, 122, 82, 0.18);
            color: var(--risk);
        }

        .shap-pill.protective {
            background: rgba(79, 168, 143, 0.18);
            color: var(--protective);
        }

        .scenario-delta {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 26px;
            font-weight: 600;
            margin: 10px 0;
        }
    """),

    ui.div(
        ui.h2("Customer Churn Risk Dashboard", class_="dashboard-title"),
        ui.p(
            "Score an individual customer, review what is driving the prediction, and test hypothetical changes.",
            class_="dashboard-subtitle"
        ),
        class_="dashboard-header"
    ),

    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.div("Customer selection", class_="sidebar-heading"),
                ui.input_text("customer_id", "Customer ID", value=default_customer),
                ui.input_action_button("analyze", "Analyze customer", class_="btn-primary w-100"),
                class_="sidebar-block"
            ),

            ui.hr(),

            ui.div(
                ui.div("What-if controls", class_="sidebar-heading"),
                ui.input_select("feature", "Feature", choices=feature_choices, selected=default_feature),
                ui.output_ui("current_value_ui"),
                ui.output_ui("scenario_input"),
                class_="sidebar-block"
            ),

            ui.hr(),

            ui.div(
                ui.div("Model information", class_="sidebar-heading"),
                ui.div("Decision threshold", class_="summary-label"),
                ui.div(
                    f"{threshold:.0%}",
                    style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;"
                )
            ),

            width=300
        ),

        ui.div(
            ui.h3("Current customer risk", class_="section-title"),

            ui.layout_columns(
                ui.div(
                    ui.div("Predicted churn probability", class_="summary-label"),
                    ui.output_ui("probability_card"),
                    class_="summary-card"
                ),
                ui.output_ui("prediction_card_wrap"),
                ui.output_ui("actual_card_wrap"),
                col_widths=[4, 4, 4]
            ),

            ui.br(),

            ui.div(
                ui.layout_columns(
                    ui.div(
                        ui.div(
                            ui.div("Individual model explanation", class_="sidebar-heading"),
                            ui.output_plot(
                                "waterfall_plot",
                                height=f"{int(96 * (0.8 + 0.29 * len(final_features)))}px"
                            ),
                            class_="panel-card"
                        ),
                        style="height:100%;"
                    ),

                    ui.div(
                        ui.div(
                            ui.div(
                                ui.div("Top churn-risk drivers", class_="sidebar-heading"),
                                ui.output_ui("risk_table"),
                                class_="driver-section"
                            ),
                            ui.div(
                                ui.div("Top protective drivers", class_="sidebar-heading"),
                                ui.output_ui("protective_table"),
                                class_="driver-section"
                            ),
                            class_="panel-card driver-panel"
                        ),
                        style="height:100%;"
                    ),

                    col_widths=[9, 3]
                ),
                class_="equal-height-row"
            ),

            ui.hr(),

            ui.h3("What-if scenario analysis", class_="section-title"),

            ui.layout_columns(
                ui.div(
                    ui.output_ui("scenario_summary"),
                    class_="panel-card"
                ),

                ui.div(
                    ui.div("Feature sensitivity", class_="sidebar-heading"),
                    ui.output_plot("what_if_plot", height="260px"),
                    class_="panel-card"
                ),

                col_widths=[4, 8]
            ),

            ui.p(
                ui.tags.em(
                    "What-if results describe model sensitivity under hypothetical inputs and do not imply causal effects."
                ),
                class_="subtle",
                style="margin-top:10px;"
            )
        )
    )
)


def server(input, output, session):

    @reactive.calc
    @reactive.event(input.analyze)
    def result():
        customer_id = input.customer_id().strip()
        return score_customer(customer_id) if customer_id else None


    @output
    @render.ui
    def current_value_ui():
        r = result()
        if r is None:
            return ui.div()

        feature = input.feature()
        value = get_current_feature_value(r["customer_x"], feature)

        if feature == "tenure":
            value = f"{value} months"

        return ui.div(
            ui.div("Current value", class_="summary-label"),
            ui.div(str(value), style="font-weight:600;margin-bottom:10px;")
        )


    @output
    @render.ui
    def scenario_input():
        feature = input.feature()

        if feature == "tenure":
            return ui.input_slider(
                "scenario_value",
                "Scenario tenure",
                min=0,
                max=tenure_max,
                value=min(24, tenure_max),
                step=1
            )

        options = get_scenario_options(feature)

        return ui.input_select(
            "scenario_value",
            "Scenario value",
            choices=options,
            selected=next(iter(options.keys()))
        )


    @reactive.calc
    def scenario_result():
        r = result()
        if r is None:
            return None

        feature = input.feature()
        raw = input.scenario_value()

        if feature == "tenure":
            scenario_value = float(raw)
        elif feature in binary_features:
            scenario_value = int(raw)
        else:
            scenario_value = raw

        scenario_x = apply_scenario(r["customer_x"], feature, scenario_value)
        scenario_x = scenario_x[final_features].astype(float)

        probability = model.predict_proba(scenario_x)[0, 1]
        prediction = "Churn" if probability >= threshold else "No Churn"

        return {
            "feature": feature,
            "scenario_value": scenario_value,
            "probability": probability,
            "prediction": prediction
        }


    @output
    @render.ui
    def probability_card():
        r = result()
        return ui.div("—", class_="summary-value") if r is None else ui.div(
            f"{r['probability']:.1%}",
            class_="summary-value"
        )


    @output
    @render.ui
    def prediction_card_wrap():
        r = result()
        accent = "var(--border)"
        value = "—"

        if r is not None:
            accent = "var(--risk)" if r["prediction"] == "Churn" else "var(--protective)"
            value = r["prediction"]

        return ui.div(
            ui.div("Model prediction", class_="summary-label"),
            ui.div(value, class_="summary-value"),
            class_="summary-card",
            style=f"border-left-color:{accent};"
        )


    @output
    @render.ui
    def actual_card_wrap():
        r = result()
        accent = "var(--border)"
        value = "—"

        if r is not None:
            accent = "var(--risk)" if r["actual"] == "Churn" else "var(--protective)"
            value = r["actual"]

        return ui.div(
            ui.div("Historical outcome", class_="summary-label"),
            ui.div(value, class_="summary-value"),
            class_="summary-card",
            style=f"border-left-color:{accent};"
        )


    @output
    @render.plot
    def waterfall_plot():
        r = result()
        if r is None:
            return

        fig = plt.figure(figsize=(12, 0.8 + 0.29 * len(final_features)))

        shap.plots.waterfall(
            r["shap_values"][0],
            max_display=len(final_features),
            show=False
        )

        fig = plt.gcf()
        ax = plt.gca()
        ax.margins(y=0.01)

        fig.patch.set_facecolor(COLOR_PANEL)
        ax.set_facecolor(COLOR_PANEL)

        plt.subplots_adjust(left=0.31, right=0.97, top=0.95, bottom=0.06)

        return fig


    @output
    @render.ui
    def risk_table():
        r = result()
        if r is None:
            return ui.div()

        t = r["risk_drivers"].copy()

        rows = [
            ui.tags.tr(
                ui.tags.td(str(row["Feature"])),
                ui.tags.td(f"{row['Customer Value']:.2f}"),
                ui.tags.td(
                    ui.tags.span(
                        f"+{row['SHAP Impact']:.2f}",
                        class_="shap-pill risk"
                    )
                )
            )
            for _, row in t.iterrows()
        ]

        return ui.tags.table(
            ui.tags.colgroup(
                ui.tags.col(class_="col-feature"),
                ui.tags.col(class_="col-value"),
                ui.tags.col(class_="col-shap"),
            ),
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Feature"),
                    ui.tags.th("Value"),
                    ui.tags.th("SHAP")
                )
            ),
            ui.tags.tbody(*rows),
            class_="driver-table"
        )


    @output
    @render.ui
    def protective_table():
        r = result()
        if r is None:
            return ui.div()

        t = r["protective_drivers"].copy()

        rows = [
            ui.tags.tr(
                ui.tags.td(str(row["Feature"])),
                ui.tags.td(f"{row['Customer Value']:.2f}"),
                ui.tags.td(
                    ui.tags.span(
                        f"{row['SHAP Impact']:.2f}",
                        class_="shap-pill protective"
                    )
                )
            )
            for _, row in t.iterrows()
        ]

        return ui.tags.table(
            ui.tags.colgroup(
                ui.tags.col(class_="col-feature"),
                ui.tags.col(class_="col-value"),
                ui.tags.col(class_="col-shap"),
            ),
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Feature"),
                    ui.tags.th("Value"),
                    ui.tags.th("SHAP")
                )
            ),
            ui.tags.tbody(*rows),
            class_="driver-table"
        )


    @output
    @render.ui
    def scenario_summary():
        r = result()
        s = scenario_result()

        if r is None or s is None:
            return ui.p("Analyze a customer to begin.", class_="subtle")

        feature = s["feature"]
        current_value = get_current_feature_value(r["customer_x"], feature)

        if feature == "tenure":
            current_display = f"{int(current_value)} months"
            scenario_display = f"{int(float(s['scenario_value']))} months"
        elif feature in binary_features:
            current_display = current_value
            scenario_display = "Yes" if int(s["scenario_value"]) == 1 else "No"
        else:
            current_display = current_value
            scenario_display = str(s["scenario_value"])

        delta = (s["probability"] - r["probability"]) * 100
        delta_color = (
            "var(--risk)"
            if delta > 0
            else "var(--protective)"
            if delta < 0
            else "var(--text-muted)"
        )

        return ui.div(
            ui.div("Current to scenario", class_="sidebar-heading"),
            ui.p(
                ui.strong(FEATURE_LABELS.get(feature, feature) + ": "),
                f"{current_display} \u2192 {scenario_display}"
            ),
            ui.div(
                f"{r['probability']:.1%} \u2192 {s['probability']:.1%}",
                class_="scenario-delta"
            ),
            ui.p(f"{r['prediction']} \u2192 {s['prediction']}"),
            ui.p(
                ui.strong("Change: "),
                ui.span(
                    f"{delta:+.1f} pp",
                    style=f"color:{delta_color};font-weight:600;"
                )
            )
        )


    @output
    @render.plot
    def what_if_plot():
        r = result()
        s = scenario_result()

        if r is None or s is None:
            return

        feature = s["feature"]
        sweep = what_if_sweep(r["customer_x"], feature)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        fig.patch.set_facecolor(COLOR_PANEL)
        ax.set_facecolor(COLOR_PANEL)

        if feature == "tenure":
            x = pd.to_numeric(sweep["Scenario Value"])
            y = sweep["Predicted Churn Probability"]

            ax.plot(x, y, marker="o", markersize=4, linewidth=2, color=COLOR_PRIMARY)

            current_tenure = float(r["customer_x"]["tenure"].iloc[0])
            scenario_tenure = float(s["scenario_value"])

            ax.axvline(
                current_tenure,
                linestyle=":",
                color=COLOR_TEXT_MUTED,
                label=f"Current ({current_tenure:.0f})"
            )

            ax.axvline(
                scenario_tenure,
                linestyle="-.",
                color=COLOR_PRIMARY,
                label=f"Scenario ({scenario_tenure:.0f})"
            )

            ax.set_xlabel("Tenure (months)")

        else:
            labels = sweep["Scenario Value"].tolist()
            probs = sweep["Predicted Churn Probability"].tolist()

            current_value = get_current_feature_value(r["customer_x"], feature)

            scenario_label = (
                "Yes" if int(s["scenario_value"]) == 1 else "No"
            ) if feature in binary_features else str(s["scenario_value"])

            bar_colors = []

            for label in labels:
                if str(label) == str(scenario_label):
                    bar_colors.append(COLOR_PRIMARY)
                elif str(label) == str(current_value):
                    bar_colors.append(COLOR_TEXT_MUTED)
                else:
                    bar_colors.append(COLOR_BORDER)

            bars = ax.bar(labels, probs, color=bar_colors)

            for bar, label in zip(bars, labels):
                notes = []

                if str(label) == str(current_value):
                    notes.append("Current")

                if str(label) == str(scenario_label):
                    notes.append("Scenario")

                if notes:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        " / ".join(notes),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=COLOR_TEXT
                    )

            ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
            ax.tick_params(axis="x", rotation=12)

        ax.axhline(
            threshold,
            linestyle="--",
            color=COLOR_RISK,
            label=f"Threshold ({threshold:.0%})"
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted churn probability")
        ax.set_title(
            f"What-if: {FEATURE_LABELS.get(feature, feature)}",
            fontsize=12,
            color=COLOR_TEXT
        )

        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

        legend = ax.legend(
            fontsize=8,
            facecolor=COLOR_PANEL,
            edgecolor=COLOR_BORDER
        )

        for text in legend.get_texts():
            text.set_color(COLOR_TEXT)

        plt.tight_layout()
        return fig


app = App(app_ui, server)
