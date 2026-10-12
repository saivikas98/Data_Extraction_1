import io
import re
from datetime import datetime

import importlib
import subprocess
import sys


def ensure_package(import_name, pip_name):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        importlib.invalidate_caches()
        return importlib.import_module(import_name)


st = ensure_package("streamlit", "streamlit>=1.38,<2")
pd = ensure_package("pandas", "pandas>=2.2,<3")
ensure_package("openpyxl", "openpyxl>=3.1,<4")
_pypdf = ensure_package("pypdf", "pypdf>=5,<7")
PdfReader = _pypdf.PdfReader

st.set_page_config(page_title="Batch Process Report Extractor", page_icon="🧪", layout="wide")

TARGET_STAGES = ["WETTING", "PREMIXING", "MIXING", "BLENDING"]
DATALOG_COLUMNS = {
    "WETTING": ["Vacuum Pressure Left (mbar)", "Vacuum Pressure Right (mbar)",
                "Loadcell Left (bar)", "Loadcell Right (bar)",
                "Servo Position Left (mm)", "Servo Position Right (mm)",
                "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "PREMIXING": ["Current Cycle", "Vacuum Pressure Left (mbar)", "Vacuum Pressure Right (mbar)",
                  "Loadcell Left (bar)", "Loadcell Right (bar)",
                  "Servo Position Left (mm)", "Servo Position Right (mm)",
                  "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "MIXING": ["Current Cycle", "Vacuum Pressure Left (mbar)", "Vacuum Pressure Right (mbar)",
               "Loadcell Left (bar)", "Loadcell Right (bar)",
               "Servo Position Left (mm)", "Servo Position Right (mm)",
               "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)"],
    "BLENDING": ["Current Cycle", "Vacuum Pressure Left (mbar)", "Vacuum Pressure Right (mbar)",
                 "Loadcell Left (bar)", "Loadcell Right (bar)",
                 "Servo Position Left (mm)", "Servo Position Right (mm)",
                 "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
                 "Measured pH", "Added Water (mL)"],
}
DATE = r"\d{2}/\d{2}/\d{4}"
TIME = r"\d{2}:\d{2}:\d{2}"
NUM = r"[-+]?\d+(?:\.\d+)?|NA"


def duration_text(start, end):
    if pd.isna(start) or pd.isna(end):
        return None
    seconds = int((end - start).total_seconds())
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}.{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.cache_data(show_spinner=False)
def extract_pages(pdf_bytes):
    """Scan the complete PDF using pure-Python pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text(extraction_mode="layout") or "")
    return pages


def parse_batch_details(pages):
    text = "\n".join(pages[:5])
    labels = ["Product name", "Batch no.", "Batch started by", "Batch ended by",
              "Batch start date & time", "Batch end date & time", "Duration (D.HH:MM:SS)"]
    result = {}
    for label in labels:
        m = re.search(rf"(?m)^\s*{re.escape(label)}\s+(.+?)\s*$", text)
        result[label] = m.group(1).strip() if m else None
    return result


def parse_overview(pages):
    """Extract all PROCESS OVERVIEW rows, including non-target operations."""
    rows = []
    active = False
    rx = re.compile(
        rf"^\s*(.*?)\s{{2,}}(.*?)\s{{2,}}({DATE}\s+{TIME})\s+({DATE}\s+{TIME})\s+(\d+)\s+(\d+)\s*$"
    )
    for page_no, text in enumerate(pages, 1):
        if "PROCESS OVERVIEW" in text:
            active = True
        if not active:
            continue
        found_on_page = 0
        for line in text.splitlines():
            m = rx.match(line)
            if m:
                start = pd.to_datetime(m.group(3), dayfirst=True, errors="coerce")
                end = pd.to_datetime(m.group(4), dayfirst=True, errors="coerce")
                rows.append({"Process Name": m.group(1).strip(), "Machine Name": m.group(2).strip(),
                             "Process Start": start, "Process End": end,
                             "Duration": duration_text(start, end), "Index": int(m.group(5)),
                             "Total": int(m.group(6)), "Source Page": page_no})
                found_on_page += 1
        # Overview is at the beginning. Stop after it has ended.
        if active and rows and found_on_page == 0 and page_no > 5:
            break
    return pd.DataFrame(rows)


def stage_from_text(text, suffix="PROCESS"):
    for stage in TARGET_STAGES:
        if re.search(rf"\b{stage}\s+{suffix}\b", text):
            return stage
    return None


def parse_process_summaries(pages):
    rows = []
    labels = ["Recipe name", "Machine", "Process start", "Process end", "Duration (D.HH:MM:SS)",
              "Process started by", "Process ended by", "API accumulator ID Left",
              "API accumulator ID Right", "Water accumulator ID Left",
              "Water accumulatorID Right", "Ph accumulator ID"]
    for page_no, text in enumerate(pages, 1):
        stage = stage_from_text(text, "PROCESS")
        if not stage or f"{stage} DATALOG" in text:
            continue
        if "BATCH CYCLE REPORT" not in text:
            continue
        row = {"Stage": stage, "Source Page": page_no}
        for label in labels:
            m = re.search(rf"(?m)^\s*{re.escape(label)}\s+(.+?)\s*$", text)
            row[label] = m.group(1).strip() if m else None
        if row.get("Machine") and row.get("Process start"):
            row["Extruder"] = extract_extruder(row["Machine"])
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ["Process start", "Process end"]:
            df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
        df = df.sort_values(["Process start", "Stage", "Machine"]).drop_duplicates(
            subset=["Stage", "Machine", "Process start", "Process end"])
    return df


def extract_extruder(machine):
    m = re.search(r"Extruder\s*(\d+)", str(machine), flags=re.I)
    return int(m.group(1)) if m else None


def parse_recipe_settings(pages, summaries):
    rows = []
    for _, s in summaries.iterrows():
        text = pages[int(s["Source Page"]) - 1]
        in_settings = False
        for line in text.splitlines():
            if "RECIPE SETTINGS" in line:
                in_settings = True
                continue
            if not in_settings:
                continue
            if "DATALOG" in line or "Printed on" in line:
                break
            m = re.match(r"^\s*(.*?)\s{2,}(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\S+)\s*$", line)
            if m and m.group(1).strip() not in {"Name", ""}:
                rows.append({"Stage": s["Stage"], "Machine": s["Machine"], "Extruder": s["Extruder"],
                             "Parameter": m.group(1).strip(), "Minimum": float(m.group(2)),
                             "Set Value": float(m.group(3)), "Maximum": float(m.group(4)),
                             "Unit": m.group(5), "Source Page": s["Source Page"]})
    return pd.DataFrame(rows)


def parse_datalogs(pages, summaries):
    """Parse every target-stage datalog row and associate it with the matching process instance."""
    frames = []
    summary_lookup = summaries.copy()
    for page_no, text in enumerate(pages, 1):
        stage = stage_from_text(text, "DATALOG")
        if not stage:
            continue
        records = []
        for line in text.splitlines():
            m = re.match(rf"^\s*({DATE})\s+({TIME})\s+(.+?)\s+((?:{NUM})(?:\s+(?:{NUM}))+)\s*$", line)
            if not m:
                continue
            timestamp = pd.to_datetime(f"{m.group(1)} {m.group(2)}", dayfirst=True, errors="coerce")
            values = m.group(4).split()
            expected = len(DATALOG_COLUMNS[stage])
            if len(values) != expected:
                continue
            converted = [None if v == "NA" else float(v) for v in values]
            record = {"Date & Time": timestamp, "Step Name": m.group(3).strip()}
            record.update(dict(zip(DATALOG_COLUMNS[stage], converted)))
            record["Source Page"] = page_no
            records.append(record)
        if not records:
            continue
        df = pd.DataFrame(records)
        midpoint = df["Date & Time"].dropna().median()
        candidates = summary_lookup[summary_lookup["Stage"].eq(stage)].copy()
        if not candidates.empty and pd.notna(midpoint):
            inside = candidates[(candidates["Process start"] <= midpoint) & (candidates["Process end"] >= midpoint)]
            chosen = inside.iloc[0] if not inside.empty else candidates.iloc[(candidates["Process start"] - midpoint).abs().argmin()]
            df.insert(0, "Extruder", chosen.get("Extruder"))
            df.insert(0, "Machine", chosen.get("Machine"))
        else:
            df.insert(0, "Extruder", None); df.insert(0, "Machine", None)
        df.insert(0, "Stage", stage)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def to_excel(batch, overview, summaries, settings, datalog):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([batch]).to_excel(writer, sheet_name="Batch Details", index=False)
        overview.to_excel(writer, sheet_name="Process Overview", index=False)
        summaries.to_excel(writer, sheet_name="Process Summary", index=False)
        settings.to_excel(writer, sheet_name="Recipe Settings", index=False)
        if datalog.empty:
            datalog.to_excel(writer, sheet_name="Datalog", index=False)
        else:
            for stage in TARGET_STAGES:
                part = datalog[datalog["Stage"].eq(stage)]
                if not part.empty:
                    part.to_excel(writer, sheet_name=f"{stage.title()} Datalog"[:31], index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.sheet_view.showGridLines = False
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for col in ws.columns:
                letter = col[0].column_letter
                width = min(max(len(str(c.value or "")) for c in col) + 2, 38)
                ws.column_dimensions[letter].width = width
    return output.getvalue()


st.title("🧪 Complete Batch Process Report Extractor")
st.caption("Scans the complete PDF and extracts the Process Overview plus Wetting, Premixing, Mixing and Blending data.")
uploaded = st.file_uploader("Upload Batch Cycle Report PDF", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.getvalue()
    with st.status("Scanning complete document...", expanded=True) as status:
        pages = extract_pages(pdf_bytes)
        st.write(f"Scanned {len(pages):,} pages")
        batch = parse_batch_details(pages)
        overview = parse_overview(pages)
        summaries = parse_process_summaries(pages)
        settings = parse_recipe_settings(pages, summaries)
        datalog = parse_datalogs(pages, summaries)
        status.update(label="Extraction completed", state="complete", expanded=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF pages", f"{len(pages):,}")
    c2.metric("Overview operations", f"{len(overview):,}")
    c3.metric("Target process runs", f"{len(summaries):,}")
    c4.metric("Datalog rows", f"{len(datalog):,}")

    st.subheader("Batch Details")
    st.dataframe(pd.DataFrame([batch]), use_container_width=True, hide_index=True)

    st.subheader("Process Overview Times")
    if overview.empty:
        st.warning("No PROCESS OVERVIEW rows were recognized.")
    else:
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.download_button("Download Process Overview CSV", overview.to_csv(index=False).encode(),
                           "process_overview.csv", "text/csv")

    st.subheader("Wetting, Premixing, Mixing and Blending Summary")
    selected_stages = st.multiselect("Process stages", TARGET_STAGES, default=TARGET_STAGES)
    selected_extruders = st.multiselect("Extruders", [1, 2, 3], default=[1, 2, 3])
    summary_view = summaries[summaries["Stage"].isin(selected_stages) & summaries["Extruder"].isin(selected_extruders)]
    preferred = ["Stage", "Extruder", "Machine", "Process start", "Process end", "Duration (D.HH:MM:SS)",
                 "Recipe name", "Process started by", "Process ended by", "Source Page"]
    st.dataframe(summary_view[[c for c in preferred if c in summary_view]], use_container_width=True, hide_index=True)

    tabs = st.tabs(["Recipe Settings", "Datalog Explorer", "Extraction Diagnostics"])
    with tabs[0]:
        view = settings[settings["Stage"].isin(selected_stages) & settings["Extruder"].isin(selected_extruders)]
        st.dataframe(view, use_container_width=True, hide_index=True)
    with tabs[1]:
        if datalog.empty:
            st.warning("No target-stage datalog rows were recognized.")
        else:
            stage = st.selectbox("Datalog stage", TARGET_STAGES)
            ext = st.selectbox("Datalog extruder", [1, 2, 3])
            dview = datalog[(datalog["Stage"] == stage) & (datalog["Extruder"] == ext)]
            step_options = sorted(dview["Step Name"].dropna().unique())
            chosen_steps = st.multiselect("Step name", step_options, default=step_options)
            dview = dview[dview["Step Name"].isin(chosen_steps)]
            st.write(f"Rows: {len(dview):,}")
            st.dataframe(dview, use_container_width=True, hide_index=True, height=520)
            st.download_button("Download filtered datalog CSV", dview.to_csv(index=False).encode(),
                               f"{stage.lower()}_extruder_{ext}.csv", "text/csv")
    with tabs[2]:
        counts = (datalog.groupby(["Stage", "Extruder"], dropna=False).size().rename("Datalog Rows").reset_index()
                  if not datalog.empty else pd.DataFrame())
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.caption("Source Page columns allow every extracted item to be traced back to the PDF.")

    excel = to_excel(batch, overview, summaries, settings, datalog)
    batch_no = batch.get("Batch no.") or "batch"
    st.download_button("⬇️ Download complete extraction as Excel", excel,
                       f"{batch_no}_complete_process_extraction.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
else:
    st.info("Upload the batch report PDF to begin.")
