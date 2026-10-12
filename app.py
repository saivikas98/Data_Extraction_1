import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Batch Process Data Extractor", page_icon="🧪", layout="wide")

STAGES = ("WETTING", "PREMIXING", "MIXING", "BLENDING")
DATE = r"\d{2}/\d{2}/\d{4}"
TIME = r"\d{2}:\d{2}:\d{2}"
NUMBER = r"[-+]?\d+(?:\.\d+)?|NA"
DATALOG_COLUMNS = {
    "WETTING": ["Vacuum Left (mbar)", "Vacuum Right (mbar)", "Loadcell Left (bar)",
                "Loadcell Right (bar)", "Servo Position Left (mm)", "Servo Position Right (mm)",
                "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "PREMIXING": ["Current Cycle", "Loadcell Left (bar)", "Loadcell Right (bar)",
                   "Servo Position Left (mm)", "Servo Position Right (mm)",
                   "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "MIXING": ["Current Cycle", "Orifice 3.8 Status", "Orifice 2 Status", "Orifice 1.5 Status",
               "Loadcell Left (bar)", "Loadcell Right (bar)", "Servo Position Left (mm)",
               "Servo Position Right (mm)", "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "BLENDING": ["Current Cycle", "Vacuum Left (mbar)", "Vacuum Right (mbar)",
                 "Loadcell Left (bar)", "Loadcell Right (bar)", "Servo Position Left (mm)",
                 "Servo Position Right (mm)", "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
                 "Measured pH", "Added Water (mL)"],
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
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text(extraction_mode="layout") or "")
        except TypeError:
            pages.append(page.extract_text() or "")
    return pages


def value_after_label(text, label):
    match = re.search(rf"(?mi)^\s*{re.escape(label)}\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def parse_batch_details(pages):
    text = "\n".join(pages[:5])
    labels = ["Product name", "Batch no.", "Batch started by", "Batch ended by",
              "Batch start date & time", "Batch end date & time", "Duration (D.HH:MM:SS)"]
    return {label: value_after_label(text, label) for label in labels}


def parse_process_overview(pages):
    rows, active, empty_pages = [], False, 0
    pattern = re.compile(
        rf"^\s*(.*?)\s{{2,}}(.*?)\s{{2,}}({DATE}\s+{TIME})\s+({DATE}\s+{TIME})\s+(\d+)\s+(\d+)\s*$"
    )
    for page_number, text in enumerate(pages, 1):
        if "PROCESS OVERVIEW" in text:
            active = True
        if not active:
            continue
        count_before = len(rows)
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            start, end = parse_dt(match.group(3)), parse_dt(match.group(4))
            rows.append({"Process Name": match.group(1).strip(), "Machine Name": match.group(2).strip(),
                         "Process Start": start, "Process End": end, "Duration": duration_text(start, end),
                         "Index": int(match.group(5)), "Total": int(match.group(6)), "Source Page": page_number})
        empty_pages = empty_pages + 1 if len(rows) == count_before else 0
        if rows and empty_pages >= 2:
            break
    return pd.DataFrame(rows)


def identify_stage(text, suffix):
    """Use exact heading lines so MIXING is not confused with PREMIXING."""
    wanted = {f"{stage} {suffix}": stage for stage in STAGES}
    for line in text.splitlines():
        heading = re.sub(r"\s+", " ", line.strip().upper())
        if heading in wanted:
            return wanted[heading]
    return None


def extruder_number(machine):
    match = re.search(r"Extruder\s*(\d+)", str(machine), re.I)
    return int(match.group(1)) if match else None


def parse_process_summaries(pages):
    labels = ["Recipe name", "Machine", "Process start", "Process end", "Duration (D.HH:MM:SS)",
              "Process started by", "Process ended by", "API accumulator ID Left",
              "API accumulator ID Right", "Water accumulator ID Left", "Water accumulatorID Right",
              "Ph accumulator ID"]
    rows = []
    for page_number, text in enumerate(pages, 1):
        stage = identify_stage(text, "PROCESS")
        if not stage or f"{stage} DATALOG" in text.upper() or "BATCH CYCLE REPORT" not in text:
            continue
        row = {"Stage": stage, "Source Page": page_number}
        row.update({label: value_after_label(text, label) for label in labels})
        if row.get("Machine") and row.get("Process start"):
            row["Extruder"] = extruder_number(row["Machine"])
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Process start"] = df["Process start"].map(parse_dt)
    df["Process end"] = df["Process end"].map(parse_dt)
    return (df.drop_duplicates(["Stage", "Machine", "Process start", "Process end"])
            .sort_values(["Process start", "Stage", "Machine"]).reset_index(drop=True))


def parse_recipe_settings(pages, summaries):
    """Extract settings even when the settings table starts on the next PDF page."""
    rows = []
    setting_pattern = re.compile(
        r"^\s*(.*?)\s{2,}(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(\S+)\s*$"
    )

    for _, summary in summaries.iterrows():
        stage = summary["Stage"]
        start_page = int(summary["Source Page"]) - 1
        active = False

        # Mixing and blending settings are printed on the page after the process summary.
        for page_index in range(start_page, min(start_page + 4, len(pages))):
            text = pages[page_index]

            if re.search(rf"(?mi)^\s*{stage}\s+RECIPE SETTINGS\s*$", text):
                active = True

            if not active:
                continue

            for line in text.splitlines():
                if "RECIPE SETTINGS" in line.upper():
                    continue
                match = setting_pattern.match(line)
                if not match:
                    continue
                parameter = match.group(1).strip()
                if not parameter or parameter.lower() == "name":
                    continue
                rows.append({
                    "Stage": stage,
                    "Extruder": summary["Extruder"],
                    "Machine": summary["Machine"],
                    "Parameter": parameter,
                    "Minimum": float(match.group(2)),
                    "Set Value": float(match.group(3)),
                    "Maximum": float(match.group(4)),
                    "Unit": match.group(5),
                    "Source Page": page_index + 1,
                })

            # Stop once the next datalog begins.
            if re.search(rf"(?mi)^\s*{stage}\s+DATALOG\s*$", text):
                break

    if not rows:
        return pd.DataFrame(columns=["Stage", "Extruder", "Machine", "Parameter", "Minimum",
                                     "Set Value", "Maximum", "Unit", "Source Page"])
    return pd.DataFrame(rows).drop_duplicates(
        ["Stage", "Extruder", "Machine", "Parameter", "Source Page"]
    ).reset_index(drop=True)


def closest_summary(stage, timestamp, summaries):
    candidates = summaries[summaries["Stage"].eq(stage)].copy()
    if candidates.empty or pd.isna(timestamp):
        return None
    inside = candidates[(candidates["Process start"] <= timestamp) & (candidates["Process end"] >= timestamp)]
    if not inside.empty:
        return inside.iloc[0]
    distances = (candidates["Process start"] - timestamp).abs()
    return candidates.loc[distances.idxmin()]


def parse_datalogs(pages, summaries):
    """Extract datalogs and report candidate, extracted and rejected rows by page."""
    frames, diagnostic_rows, rejected_rows = [], [], []
    timestamp_line = re.compile(rf"^\s*({DATE})\s+({TIME})\s+(.+?)\s*$")

    for page_number, text in enumerate(pages, 1):
        stage = identify_stage(text, "DATALOG")
        if not stage:
            continue

        expected = len(DATALOG_COLUMNS[stage])
        row_pattern = re.compile(
            rf"^\s*({DATE})\s+({TIME})\s+(.+?)\s+((?:\S+\s+){{{expected - 1}}}\S+)\s*$"
        )
        records = []
        candidate_count = 0
        rejected_count = 0

        for line_number, line in enumerate(text.splitlines(), 1):
            if not timestamp_line.match(line):
                continue
            candidate_count += 1
            match = row_pattern.match(line)
            if not match:
                rejected_count += 1
                rejected_rows.append({
                    "Stage": stage, "Source Page": page_number, "Line Number": line_number,
                    "Reason": f"Could not isolate {expected} trailing fields",
                    "Raw Line": line.strip(),
                })
                continue

            raw_values = match.group(4).split()
            if len(raw_values) != expected:
                rejected_count += 1
                rejected_rows.append({
                    "Stage": stage, "Source Page": page_number, "Line Number": line_number,
                    "Reason": f"Expected {expected} fields; found {len(raw_values)}",
                    "Raw Line": line.strip(),
                })
                continue

            values = []
            for value in raw_values:
                if value.upper() == "NA":
                    values.append(None)
                else:
                    try:
                        values.append(float(value))
                    except ValueError:
                        values.append(value)

            record = {
                "Date & Time": parse_dt(f"{match.group(1)} {match.group(2)}"),
                "Step Name": match.group(3).strip(), "Source Page": page_number,
            }
            record.update(dict(zip(DATALOG_COLUMNS[stage], values)))
            records.append(record)

        diagnostic_rows.append({
            "Stage": stage, "Source Page": page_number,
            "Expected Fields": expected, "Candidate Timestamp Rows": candidate_count,
            "Extracted Rows": len(records), "Rejected/Missing Rows": rejected_count,
            "Extraction Rate (%)": round(100 * len(records) / candidate_count, 2) if candidate_count else 100.0,
            "Status": "OK" if rejected_count == 0 else "REVIEW",
        })

        if not records:
            continue
        frame = pd.DataFrame(records)
        selected = closest_summary(stage, frame["Date & Time"].median(), summaries)
        machine = selected["Machine"] if selected is not None else None
        extruder = selected["Extruder"] if selected is not None else None
        frame.insert(0, "Extruder", extruder)
        frame.insert(0, "Machine", machine)
        frame.insert(0, "Stage", stage)
        frames.append(frame)

    datalog = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True) if frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    rejected = pd.DataFrame(rejected_rows)
    return datalog, diagnostics, rejected


def make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([batch]).to_excel(writer, "Batch Details", index=False)
        overview.to_excel(writer, "Process Overview", index=False)
        summaries.to_excel(writer, "Process Summary", index=False)
        settings.to_excel(writer, "Recipe Settings", index=False)
        diagnostics.to_excel(writer, "Datalog Diagnostics", index=False)
        rejected.to_excel(writer, "Rejected Rows", index=False)
        for stage in STAGES:
            part = datalog[datalog["Stage"].eq(stage)] if not datalog.empty else pd.DataFrame()
            part.to_excel(writer, f"{stage.title()} Datalog"[:31], index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            for cell in sheet[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for column in sheet.columns:
                letter = column[0].column_letter
                sheet.column_dimensions[letter].width = min(max(len(str(c.value or "")) for c in column) + 2, 40)
    return output.getvalue()


st.title("🧪 Complete Batch Process Report Extractor")
st.caption("Scans the complete PDF and extracts Process Overview, Wetting, Premixing, Mixing and Blending data.")
uploaded = st.file_uploader("Upload Batch Cycle Report PDF", type="pdf")

if uploaded is None:
    st.info("Upload a batch-report PDF to begin.")
    st.stop()

try:
    with st.status("Scanning the complete document...", expanded=True) as status:
        pages = extract_pages(uploaded.getvalue())
        st.write(f"Pages scanned: {len(pages):,}")
        batch = parse_batch_details(pages)
        overview = parse_process_overview(pages)
        summaries = parse_process_summaries(pages)
        settings = parse_recipe_settings(pages, summaries)
        datalog, datalog_diagnostics, rejected_rows = parse_datalogs(pages, summaries)
        status.update(label="Extraction completed", state="complete", expanded=False)
except Exception as error:
    st.exception(error)
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("PDF pages", f"{len(pages):,}")
m2.metric("Overview operations", f"{len(overview):,}")
m3.metric("Target process runs", f"{len(summaries):,}")
m4.metric("Datalog rows", f"{len(datalog):,}")

st.subheader("Batch Details")
st.dataframe(pd.DataFrame([batch]), use_container_width=True, hide_index=True)

st.subheader("Process Overview Times")
st.dataframe(overview, use_container_width=True, hide_index=True)

st.subheader("Wetting, Premixing, Mixing and Blending Summary")
selected_stages = st.multiselect("Stages", STAGES, default=list(STAGES))
extruder_values = sorted(int(x) for x in summaries.get("Extruder", pd.Series(dtype=float)).dropna().unique())
selected_extruders = st.multiselect("Extruders", extruder_values, default=extruder_values)
summary_view = summaries[summaries["Stage"].isin(selected_stages)]
if selected_extruders:
    summary_view = summary_view[summary_view["Extruder"].isin(selected_extruders)]
st.dataframe(summary_view, use_container_width=True, hide_index=True)

recipe_tab, data_tab, diagnostics_tab = st.tabs(["Recipe Settings", "Datalog Explorer", "Diagnostics"])
with recipe_tab:
    recipe_view = settings[settings["Stage"].isin(selected_stages)] if not settings.empty else settings
    st.dataframe(recipe_view, use_container_width=True, hide_index=True)
with data_tab:
    if datalog.empty:
        st.warning("No datalog records were recognized.")
    else:
        stage = st.selectbox("Datalog stage", STAGES)
        available = sorted(int(x) for x in datalog.loc[datalog["Stage"].eq(stage), "Extruder"].dropna().unique())
        extruder = st.selectbox("Datalog extruder", available) if available else None
        view = datalog[datalog["Stage"].eq(stage)]
        if extruder is not None:
            view = view[view["Extruder"].eq(extruder)]
        steps = sorted(view["Step Name"].dropna().unique())
        chosen = st.multiselect("Step names", steps, default=steps)
        view = view[view["Step Name"].isin(chosen)]
        st.write(f"Rows: {len(view):,}")
        st.dataframe(view, use_container_width=True, hide_index=True, height=520)
        st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                           f"{stage.lower()}_extruder_{extruder}.csv", "text/csv")
with diagnostics_tab:
    st.markdown("#### Stage-level extraction audit")
    if datalog_diagnostics.empty:
        st.warning("No datalog pages were detected in the PDF.")
    else:
        stage_audit = (datalog_diagnostics.groupby("Stage", as_index=False)
                       .agg(**{"Datalog Pages": ("Source Page", "nunique"),
                               "Candidate Timestamp Rows": ("Candidate Timestamp Rows", "sum"),
                               "Extracted Rows": ("Extracted Rows", "sum"),
                               "Rejected/Missing Rows": ("Rejected/Missing Rows", "sum")}))
        stage_audit["Extraction Rate (%)"] = (
            100 * stage_audit["Extracted Rows"] / stage_audit["Candidate Timestamp Rows"].replace(0, pd.NA)
        ).round(2).fillna(100.0)
        stage_audit["Status"] = stage_audit["Rejected/Missing Rows"].map(lambda x: "OK" if x == 0 else "REVIEW")
        st.dataframe(stage_audit, use_container_width=True, hide_index=True)

        missing_stages = [stage for stage in STAGES if stage not in set(stage_audit["Stage"])]
        if missing_stages:
            st.error("No datalog pages detected for: " + ", ".join(missing_stages))

        total_candidates = int(stage_audit["Candidate Timestamp Rows"].sum())
        total_extracted = int(stage_audit["Extracted Rows"].sum())
        total_rejected = int(stage_audit["Rejected/Missing Rows"].sum())
        d1, d2, d3 = st.columns(3)
        d1.metric("Timestamp rows found", f"{total_candidates:,}")
        d2.metric("Rows extracted", f"{total_extracted:,}")
        d3.metric("Rows requiring review", f"{total_rejected:,}")

        st.markdown("#### Page-level diagnostics")
        review_only = st.checkbox("Show only pages with missing/rejected rows", value=True)
        page_view = datalog_diagnostics
        if review_only:
            page_view = page_view[page_view["Rejected/Missing Rows"] > 0]
        st.dataframe(page_view, use_container_width=True, hide_index=True, height=360)

    st.markdown("#### Rejected row details")
    if rejected_rows.empty:
        st.success("No timestamp rows were rejected by the parser.")
    else:
        st.warning(f"{len(rejected_rows):,} timestamp rows require review.")
        st.dataframe(rejected_rows, use_container_width=True, hide_index=True, height=360)
        st.download_button("Download rejected rows CSV", rejected_rows.to_csv(index=False).encode("utf-8"),
                           "rejected_datalog_rows.csv", "text/csv")

    st.markdown("#### Recipe-setting coverage")
    expected_runs = summaries.groupby("Stage").size().rename("Process Runs").reset_index() if not summaries.empty else pd.DataFrame()
    extracted_settings = settings.groupby("Stage").size().rename("Setting Rows").reset_index() if not settings.empty else pd.DataFrame()
    if not expected_runs.empty:
        recipe_audit = expected_runs.merge(extracted_settings, on="Stage", how="left").fillna({"Setting Rows": 0})
        recipe_audit["Setting Rows"] = recipe_audit["Setting Rows"].astype(int)
        recipe_audit["Status"] = recipe_audit["Setting Rows"].map(lambda x: "OK" if x > 0 else "MISSING")
        st.dataframe(recipe_audit, use_container_width=True, hide_index=True)

    st.caption("Candidate rows are lines beginning with a valid date and time on detected datalog pages. Rejected rows are retained with the PDF page, line number, reason and raw text for investigation.")

excel = make_excel(batch, overview, summaries, settings, datalog, datalog_diagnostics, rejected_rows)
batch_number = batch.get("Batch no.") or "batch"
st.download_button("⬇️ Download complete Excel extraction", excel,
                   f"{batch_number}_complete_process_extraction.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
