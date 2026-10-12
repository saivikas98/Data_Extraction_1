import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Batch Process Data Extractor", page_icon="🧪", layout="wide")

STAGES = ("WETTING", "PREMIXING", "MIXING", "BLENDING")
DATE = r"\d{2}/\d{2}/\d{4}"
TIME = r"\d{2}:\d{2}:\d{2}"
DATALOG_COLUMNS = {
    "WETTING": ["Vacuum Left (mbar)", "Vacuum Right (mbar)", "Loadcell Left (bar)", "Loadcell Right (bar)", "Servo Position Left (mm)", "Servo Position Right (mm)", "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "PREMIXING": ["Current Cycle", "Loadcell Left (bar)", "Loadcell Right (bar)", "Servo Position Left (mm)", "Servo Position Right (mm)", "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "MIXING": ["Current Cycle", "Orifice 3.8 Status", "Orifice 2 Status", "Orifice 1.5 Status", "Loadcell Left (bar)", "Loadcell Right (bar)", "Servo Position Left (mm)", "Servo Position Right (mm)", "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "BLENDING": ["Current Cycle", "Vacuum Left (mbar)", "Vacuum Right (mbar)", "Loadcell Left (bar)", "Loadcell Right (bar)", "Servo Position Left (mm)", "Servo Position Right (mm)", "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)", "Measured pH", "Added Water (mL)"],
}


def parse_dt(value):
    return pd.to_datetime(value, format="%d/%m/%Y %H:%M:%S", errors="coerce")


def duration_text(start, end):
    if pd.isna(start) or pd.isna(end):
        return None
    seconds = max(0, int((end - start).total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}.{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.cache_data(show_spinner=False)
def extract_pages(pdf_bytes):
    parser_version = "2026.10.12-wetting-rise-time-v1"
    _ = parser_version
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout")
        except TypeError:
            text = page.extract_text()
        pages.append(text or "")
    return pages


def value_after_label(text, label):
    match = re.search(rf"(?mi)^\s*{re.escape(label)}\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def identify_stage(text, suffix):
    for stage in STAGES:
        if re.search(rf"(?mi)^\s*{stage}\s+{suffix}\s*$", text):
            return stage
    return None


def extruder_number(machine):
    match = re.search(r"Extruder\s*(\d+)", str(machine), re.I)
    return int(match.group(1)) if match else None


def parse_batch_details(pages):
    text = "\n".join(pages[:5])
    labels = ["Product name", "Batch no.", "Batch started by", "Batch ended by", "Batch start date & time", "Batch end date & time", "Duration (D.HH:MM:SS)"]
    return {label: value_after_label(text, label) for label in labels}


def parse_process_overview(pages):
    rows, active, empty_pages = [], False, 0
    pattern = re.compile(rf"^\s*(.*?)\s{{2,}}(.*?)\s{{2,}}({DATE}\s+{TIME})\s+({DATE}\s+{TIME})\s+(\d+)\s+(\d+)\s*$")
    for page_number, text in enumerate(pages, 1):
        if "PROCESS OVERVIEW" in text.upper(): active = True
        if not active: continue
        before = len(rows)
        for line in text.splitlines():
            match = pattern.match(line)
            if not match: continue
            start, end = parse_dt(match.group(3)), parse_dt(match.group(4))
            rows.append({"Process Name": match.group(1).strip(), "Machine Name": match.group(2).strip(), "Process Start": start, "Process End": end, "Duration": duration_text(start, end), "Index": int(match.group(5)), "Total": int(match.group(6)), "Source Page": page_number})
        empty_pages = empty_pages + 1 if len(rows) == before else 0
        if rows and empty_pages >= 2: break
    return pd.DataFrame(rows)


def parse_process_summaries(pages):
    labels = ["Recipe name", "Machine", "Process start", "Process end", "Duration (D.HH:MM:SS)", "Process started by", "Process ended by", "API accumulator ID Left", "API accumulator ID Right", "Water accumulator ID Left", "Water accumulatorID Right", "Ph accumulator ID"]
    rows = []
    for page_number, text in enumerate(pages, 1):
        stage = identify_stage(text, "PROCESS")
        if not stage or "BATCH CYCLE REPORT" not in text.upper(): continue
        row = {"Stage": stage, "Source Page": page_number}
        row.update({label: value_after_label(text, label) for label in labels})
        if row.get("Machine") and row.get("Process start"):
            row["Extruder"] = extruder_number(row["Machine"]); rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty: return frame
    frame["Process start"] = frame["Process start"].map(parse_dt)
    frame["Process end"] = frame["Process end"].map(parse_dt)
    return frame.drop_duplicates(["Stage", "Machine", "Process start", "Process end"]).sort_values("Source Page").reset_index(drop=True)


def parse_recipe_settings(pages, summaries):
    rows = []
    pattern = re.compile(r"^\s*(.*?)\s{2,}(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\S+)\s*$")
    for _, summary in summaries.iterrows():
        stage, first, active = summary["Stage"], int(summary["Source Page"]) - 1, False
        for page_index in range(first, min(first + 5, len(pages))):
            text = pages[page_index]
            if re.search(rf"(?mi)^\s*{stage}\s+RECIPE SETTINGS\s*$", text): active = True
            if not active: continue
            for line in text.splitlines():
                if "RECIPE SETTINGS" in line.upper(): continue
                match = pattern.match(line)
                if not match: continue
                parameter = match.group(1).strip()
                if not parameter or parameter.lower() == "name": continue
                rows.append({"Stage": stage, "Extruder": summary["Extruder"], "Machine": summary["Machine"], "Parameter": parameter, "Minimum": float(match.group(2)), "Set Value": float(match.group(3)), "Maximum": float(match.group(4)), "Unit": match.group(5), "Source Page": page_index + 1})
            if re.search(rf"(?mi)^\s*{stage}\s+DATALOG\s*$", text): break
    columns = ["Stage", "Extruder", "Machine", "Parameter", "Minimum", "Set Value", "Maximum", "Unit", "Source Page"]
    return pd.DataFrame(rows).drop_duplicates(["Stage", "Extruder", "Machine", "Parameter", "Source Page"]).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)


def parse_datalogs(pages, summaries):
    extracted_frames, diagnostic_rows, rejected_rows, process_ranges = [], [], [], []
    timestamp_start = re.compile(rf"^\s*({DATE})\s+({TIME})\s+")
    for _, summary in summaries.sort_values("Source Page").reset_index(drop=True).iterrows():
        stage, machine, extruder = summary["Stage"], summary["Machine"], summary["Extruder"]
        summary_page, expected = int(summary["Source Page"]), len(DATALOG_COLUMNS[stage])
        row_pattern = re.compile(rf"^\s*({DATE})\s+({TIME})\s+(.+?)\s+((?:\S+\s+){{{expected - 1}}}\S+)\s*$")
        started = False; first_page = None; last_page = None; run_frames = []
        terminal_found = False; terminal_name = None
        for page_number in range(summary_page, len(pages) + 1):
            text = pages[page_number - 1]
            heading = bool(re.search(rf"(?mi)^\s*{stage}\s+DATALOG\s*$", text))
            if not started:
                if not heading:
                    if page_number > summary_page and identify_stage(text, "PROCESS"): break
                    continue
                started = True; first_page = page_number
            elif not heading: break
            last_page = page_number; records = []; candidates = 0; rejected = 0
            for line_number, line in enumerate(text.splitlines(), 1):
                if not timestamp_start.match(line): continue
                candidates += 1; match = row_pattern.match(line)
                if not match:
                    rejected += 1; rejected_rows.append({"Stage": stage, "Machine": machine, "Extruder": extruder, "Source Page": page_number, "Line Number": line_number, "Reason": f"Could not isolate {expected} trailing fields", "Raw Line": line.strip()}); continue
                raw = match.group(4).split()
                if len(raw) != expected:
                    rejected += 1; continue
                values = []
                for value in raw:
                    if value.upper() == "NA": values.append(None)
                    else:
                        try: values.append(float(value))
                        except ValueError: values.append(value)
                record = {"Stage": stage, "Machine": machine, "Extruder": extruder, "Date & Time": parse_dt(f"{match.group(1)} {match.group(2)}"), "Step Name": match.group(3).strip(), "Source Page": page_number, "Source Line": line_number}
                record.update(dict(zip(DATALOG_COLUMNS[stage], values))); records.append(record)
                normalized = re.sub(r"\s+", " ", record["Step Name"]).strip().casefold()
                if stage == "BLENDING" and normalized in {"blending is done", "blending is completed", "blending completed"}:
                    terminal_found = True; terminal_name = record["Step Name"]; break
            diagnostic_rows.append({"Stage": stage, "Machine": machine, "Extruder": extruder, "Summary Page": summary_page, "Source Page": page_number, "Expected Fields": expected, "Candidate Timestamp Rows": candidates, "Extracted Rows": len(records), "Rejected/Missing Rows": rejected, "Extraction Rate (%)": round(100 * len(records) / candidates, 2) if candidates else 100.0, "Status": "OK" if rejected == 0 else "REVIEW"})
            if records: run_frames.append(pd.DataFrame(records))
            if terminal_found: break
        process_ranges.append({"Stage": stage, "Machine": machine, "Extruder": extruder, "Summary Page": summary_page, "First Datalog Page": first_page, "Last Datalog Page": last_page, "Datalog Pages": last_page - first_page + 1 if first_page and last_page else 0, "Terminal Step Found": terminal_found, "Terminal Step Name": terminal_name, "Status": "OK" if first_page and (stage != "BLENDING" or terminal_found) else "DATALOG NOT FOUND" if not first_page else "TERMINAL STEP NOT FOUND"})
        if run_frames: extracted_frames.append(pd.concat(run_frames, ignore_index=True))
    datalog = pd.concat(extracted_frames, ignore_index=True).drop_duplicates().sort_values(["Stage", "Extruder", "Source Page", "Source Line"], kind="stable").reset_index(drop=True) if extracted_frames else pd.DataFrame()
    if not datalog.empty:
        preferred = ["Stage", "Machine", "Extruder", "Date & Time", "Step Name", "Source Page", "Source Line"]
        datalog = datalog[preferred + [c for c in datalog.columns if c not in preferred]]
    return datalog, pd.DataFrame(diagnostic_rows), pd.DataFrame(rejected_rows), pd.DataFrame(process_ranges)


def build_wetting_rise_time_table(wetting_data):
    columns = ["Extruder", "Date & Time", "Step Name", "Time taken to reach vacuum rise (min)", "Rise time (Sec)", "Vacuum Timer Start", "Source Page"]
    if wetting_data.empty: return pd.DataFrame(columns=columns)
    work = wetting_data.copy()
    work["Normalized Step"] = work["Step Name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    work = work[work["Normalized Step"].isin({"start timeout timer", "vacuum timers", "stop vacuum timer", "revacuum"})]
    work = work.sort_values(["Extruder", "Source Page", "Source Line"], kind="stable").drop_duplicates(["Extruder", "Date & Time", "Normalized Step"])
    output = []
    for extruder, group in work.groupby("Extruder", sort=True):
        starts = group[group["Normalized Step"].eq("start timeout timer")]
        if starts.empty: continue
        time_zero = starts.iloc[0]["Date & Time"]; last_timer = None
        for _, row in group.iterrows():
            event_time, step = row["Date & Time"], row["Normalized Step"]
            if pd.isna(event_time) or event_time < time_zero: continue
            rise, paired_start = pd.NA, pd.NaT
            if step == "vacuum timers": last_timer = event_time
            elif step == "stop vacuum timer" and last_timer is not None and last_timer <= event_time:
                rise = (event_time - last_timer).total_seconds(); paired_start = last_timer
            output.append({"Extruder": int(extruder), "Date & Time": event_time, "Step Name": row["Step Name"], "Time taken to reach vacuum rise (min)": (event_time - time_zero).total_seconds() / 60, "Rise time (Sec)": rise, "Vacuum Timer Start": paired_start, "Source Page": row["Source Page"]})
    result = pd.DataFrame(output, columns=columns)
    if not result.empty: result["Rise time (Sec)"] = pd.to_numeric(result["Rise time (Sec)"], errors="coerce")
    return result


def create_wetting_rise_time_plot(table, batch_number, limit):
    """Create an interactive Plotly chart with point-level hover details."""
    normalized_steps = (
        table["Step Name"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    plot_data = table[normalized_steps.eq("stop vacuum timer")].dropna(
        subset=["Time taken to reach vacuum rise (min)", "Rise time (Sec)"]
    ).copy()

    figure = go.Figure()
    styles = {
        1: {"symbol": "triangle-up", "color": "#9BBB59", "label": "EXTRUDER - 01"},
        2: {"symbol": "x", "color": "#35A9D6", "label": "EXTRUDER - 02"},
        3: {"symbol": "circle", "color": "#4F81BD", "label": "EXTRUDER - 03"},
    }

    for extruder in sorted(plot_data["Extruder"].dropna().unique()):
        values = plot_data[plot_data["Extruder"].eq(extruder)].sort_values(
            "Time taken to reach vacuum rise (min)", kind="stable"
        )
        style = styles.get(
            int(extruder),
            {"symbol": "circle", "color": "#666666", "label": f"EXTRUDER - {int(extruder):02d}"},
        )
        custom_data = values[[
            "Date & Time", "Step Name", "Vacuum Timer Start", "Source Page"
        ]].copy()
        custom_data["Date & Time"] = custom_data["Date & Time"].dt.strftime("%d-%m-%Y %H:%M:%S")
        custom_data["Vacuum Timer Start"] = custom_data["Vacuum Timer Start"].dt.strftime("%d-%m-%Y %H:%M:%S")

        figure.add_trace(go.Scatter(
            x=values["Time taken to reach vacuum rise (min)"],
            y=values["Rise time (Sec)"],
            mode="markers",
            name=style["label"],
            customdata=custom_data.to_numpy(),
            marker={
                "symbol": style["symbol"],
                "color": style["color"],
                "size": 9,
                "line": {"width": 1, "color": style["color"]},
            },
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date & Time: %{customdata[0]}<br>"
                "Step Name: %{customdata[1]}<br>"
                "Elapsed time: %{x:.3f} min<br>"
                "Rise time: %{y:.0f} sec<br>"
                "Vacuum Timer Start: %{customdata[2]}<br>"
                "Source Page: %{customdata[3]}"
                "<extra></extra>"
            ),
        ))

    x_max = max(
        10.0,
        float(plot_data["Time taken to reach vacuum rise (min)"].max()) * 1.08
        if not plot_data.empty else 10.0,
    )
    figure.add_hline(
        y=limit,
        line_color="#C0504D",
        line_width=3,
        annotation_text="RISE TIME LIMIT",
        annotation_position="top right",
        annotation_font_color="#C0504D",
    )
    figure.update_layout(
        title={"text": str(batch_number), "x": 0.5, "xanchor": "center"},
        xaxis_title="Time taken to reach vacuum rise (min)",
        yaxis_title="Rise time (Sec)",
        xaxis={"range": [0, x_max], "showgrid": True},
        yaxis={"range": [0, max(700.0, limit * 1.15)], "showgrid": True},
        hovermode="closest",
        legend={"x": 0.72, "y": 0.45, "bgcolor": "rgba(255,255,255,0.65)"},
        height=700,
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
    )
    return figure


def figure_to_png(fig):
    buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight"); return buffer.getvalue()


def make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected, process_ranges, wetting_rise_table):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([batch]).to_excel(writer, "Batch Details", index=False)
        overview.to_excel(writer, "Process Overview", index=False); summaries.to_excel(writer, "Process Summary", index=False)
        settings.to_excel(writer, "Recipe Settings", index=False); process_ranges.to_excel(writer, "Process Page Ranges", index=False)
        wetting_rise_table.to_excel(writer, "Wetting Rise Time", index=False)
        diagnostics.to_excel(writer, "Datalog Diagnostics", index=False); rejected.to_excel(writer, "Rejected Rows", index=False)
        for stage in STAGES:
            (datalog[datalog["Stage"].eq(stage)] if not datalog.empty else pd.DataFrame()).to_excel(writer, f"{stage.title()} Datalog"[:31], index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"; ws.sheet_view.showGridLines = False; ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF"); cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = min(max(len(str(cell.value or "")) for cell in column) + 2, 40)
    return output.getvalue()


def main():
    st.title("🧪 Complete Batch Process Report Extractor")
    st.caption("Extracts Process Overview, recipe settings, complete datalogs, diagnostics, and Wetting rise-time analysis.")
    uploaded = st.file_uploader("Upload Batch Cycle Report PDF", type=["pdf"])
    if uploaded is None: st.info("Upload a batch-report PDF to begin."); st.stop()
    try:
        with st.status("Scanning the complete document...", expanded=True) as status:
            pages = extract_pages(uploaded.getvalue()); batch = parse_batch_details(pages); overview = parse_process_overview(pages)
            summaries = parse_process_summaries(pages); settings = parse_recipe_settings(pages, summaries)
            datalog, diagnostics, rejected_rows, process_ranges = parse_datalogs(pages, summaries)
            wetting_data = datalog[datalog["Stage"].eq("WETTING")].copy() if not datalog.empty else pd.DataFrame()
            wetting_rise_table = build_wetting_rise_time_table(wetting_data)
            status.update(label="Extraction completed", state="complete", expanded=False)
    except Exception as error: st.exception(error); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF pages", f"{len(pages):,}"); c2.metric("Overview operations", f"{len(overview):,}")
    c3.metric("Target process runs", f"{len(summaries):,}"); c4.metric("Datalog rows", f"{len(datalog):,}")
    st.subheader("Batch Details"); st.dataframe(pd.DataFrame([batch]), use_container_width=True, hide_index=True)
    st.subheader("Process Overview Times"); st.dataframe(overview, use_container_width=True, hide_index=True)
    st.subheader("Process Summary"); st.dataframe(summaries, use_container_width=True, hide_index=True)

    recipe_tab, datalog_tab, wetting_tab, diagnostics_tab = st.tabs(["Recipe Settings", "Datalog Explorer", "Wetting Rise-Time Plot", "Diagnostics"])
    with recipe_tab:
        st.dataframe(settings, use_container_width=True, hide_index=True, height=520)
        st.download_button("Download Recipe Settings CSV", settings.to_csv(index=False).encode(), "recipe_settings.csv", "text/csv")
    with datalog_tab:
        if datalog.empty: st.warning("No datalog rows were extracted.")
        else:
            stage = st.selectbox("Stage", [x for x in STAGES if x in set(datalog["Stage"])])
            stage_data = datalog[datalog["Stage"].eq(stage)]; extruders = sorted(stage_data["Extruder"].dropna().astype(int).unique())
            extruder = st.selectbox("Extruder", extruders) if len(extruders) else None
            view = stage_data[stage_data["Extruder"].eq(extruder)] if extruder is not None else stage_data
            steps = sorted(view["Step Name"].dropna().unique()); selected_steps = st.multiselect("Step Name", steps, default=steps)
            view = view[view["Step Name"].isin(selected_steps)].copy()
            stage_columns = ["Stage", "Machine", "Extruder", "Date & Time", "Step Name", "Source Page"] + DATALOG_COLUMNS[stage]
            view = view[[c for c in stage_columns if c in view.columns]].sort_values(["Date & Time", "Source Page"], kind="stable").reset_index(drop=True)
            view.insert(0, "Record No.", range(1, len(view) + 1)); st.dataframe(view, use_container_width=True, hide_index=True, height=560)
            st.download_button("Download Filtered Datalog CSV", view.to_csv(index=False).encode(), f"{stage.lower()}_extruder_{extruder}.csv", "text/csv")
    with wetting_tab:
        st.markdown("#### Wetting Vacuum Rise-Time Analysis")
        st.caption("Elapsed time starts at the first 'Start Timeout timer'. Rise time is calculated from each 'Vacuum Timers' event to the following 'Stop vacuum timer' event.")
        limit = st.number_input("Rise time limit (Sec)", min_value=0.0, value=600.0, step=10.0)
        if wetting_rise_table.empty: st.warning("No Wetting timer-event table was generated.")
        else:
            available = sorted(wetting_rise_table["Extruder"].unique()); selected = st.multiselect("Extruders", available, default=available)
            table_view = wetting_rise_table[wetting_rise_table["Extruder"].isin(selected)].copy()
            st.dataframe(table_view, use_container_width=True, hide_index=True, height=430)
            batch_number = batch.get("Batch no.") or "Batch"
            interactive_figure = create_wetting_rise_time_plot(
                table_view, batch_number, limit
            )
            st.plotly_chart(
                interactive_figure,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": f"{batch_number}_wetting_rise_time",
                        "height": 700,
                        "width": 1200,
                        "scale": 2,
                    },
                },
            )
            st.caption(
                "Move the cursor over any point to view Extruder, timestamp, Step Name, "
                "elapsed time, rise time, paired Vacuum Timer timestamp, and source page. "
                "Use the chart toolbar to zoom, pan, reset, or download a PNG."
            )
            interactive_html = pio.to_html(
                interactive_figure, include_plotlyjs="cdn", full_html=True
            ).encode("utf-8")
            x1, x2 = st.columns(2)
            x1.download_button(
                "Download Rise-Time Table CSV",
                table_view.to_csv(index=False).encode(),
                f"{batch_number}_wetting_rise_time.csv",
                "text/csv",
            )
            x2.download_button(
                "Download Interactive Plot HTML",
                interactive_html,
                f"{batch_number}_wetting_rise_time_interactive.html",
                "text/html",
            )
    with diagnostics_tab:
        st.markdown("#### Assigned process page ranges"); st.dataframe(process_ranges, use_container_width=True, hide_index=True)
        st.markdown("#### Page-level diagnostics"); st.dataframe(diagnostics, use_container_width=True, hide_index=True, height=400)
        st.markdown("#### Rejected rows")
        if rejected_rows.empty: st.success("No timestamp rows were rejected.")
        else: st.dataframe(rejected_rows, use_container_width=True, hide_index=True, height=400)

    excel = make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected_rows, process_ranges, wetting_rise_table)
    batch_number = batch.get("Batch no.") or "batch"
    st.download_button("⬇️ Download Complete Excel Extraction", excel, f"{batch_number}_complete_process_extraction.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")


if __name__ == "__main__":
    main()
