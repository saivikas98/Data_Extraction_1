import io
import re

import pandas as pd
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Batch Process Data Extractor", page_icon="🧪", layout="wide")

STAGES = ("WETTING", "PREMIXING", "MIXING", "BLENDING")
DATE = r"\d{2}/\d{2}/\d{4}"
TIME = r"\d{2}:\d{2}:\d{2}"

DATALOG_COLUMNS = {
    "WETTING": [
        "Vacuum Left (mbar)", "Vacuum Right (mbar)",
        "Loadcell Left (bar)", "Loadcell Right (bar)",
        "Servo Position Left (mm)", "Servo Position Right (mm)",
        "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
    ],
    "PREMIXING": [
        "Current Cycle", "Loadcell Left (bar)", "Loadcell Right (bar)",
        "Servo Position Left (mm)", "Servo Position Right (mm)",
        "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
    ],
    "MIXING": [
        "Current Cycle", "Orifice 3.8 Status", "Orifice 2 Status",
        "Orifice 1.5 Status", "Loadcell Left (bar)", "Loadcell Right (bar)",
        "Servo Position Left (mm)", "Servo Position Right (mm)",
        "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
    ],
    "BLENDING": [
        "Current Cycle", "Vacuum Left (mbar)", "Vacuum Right (mbar)",
        "Loadcell Left (bar)", "Loadcell Right (bar)",
        "Servo Position Left (mm)", "Servo Position Right (mm)",
        "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
        "Measured pH", "Added Water (mL)",
    ],
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
    labels = [
        "Product name", "Batch no.", "Batch started by", "Batch ended by",
        "Batch start date & time", "Batch end date & time", "Duration (D.HH:MM:SS)",
    ]
    return {label: value_after_label(text, label) for label in labels}


def parse_process_overview(pages):
    rows, active, empty_pages = [], False, 0
    pattern = re.compile(
        rf"^\s*(.*?)\s{{2,}}(.*?)\s{{2,}}({DATE}\s+{TIME})\s+"
        rf"({DATE}\s+{TIME})\s+(\d+)\s+(\d+)\s*$"
    )
    for page_number, text in enumerate(pages, 1):
        if "PROCESS OVERVIEW" in text.upper():
            active = True
        if not active:
            continue
        before = len(rows)
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            start, end = parse_dt(match.group(3)), parse_dt(match.group(4))
            rows.append({
                "Process Name": match.group(1).strip(),
                "Machine Name": match.group(2).strip(),
                "Process Start": start, "Process End": end,
                "Duration": duration_text(start, end),
                "Index": int(match.group(5)), "Total": int(match.group(6)),
                "Source Page": page_number,
            })
        empty_pages = empty_pages + 1 if len(rows) == before else 0
        if rows and empty_pages >= 2:
            break
    return pd.DataFrame(rows)


def parse_process_summaries(pages):
    labels = [
        "Recipe name", "Machine", "Process start", "Process end",
        "Duration (D.HH:MM:SS)", "Process started by", "Process ended by",
        "API accumulator ID Left", "API accumulator ID Right",
        "Water accumulator ID Left", "Water accumulatorID Right", "Ph accumulator ID",
    ]
    rows = []
    for page_number, text in enumerate(pages, 1):
        stage = identify_stage(text, "PROCESS")
        if not stage or "BATCH CYCLE REPORT" not in text.upper():
            continue
        row = {"Stage": stage, "Source Page": page_number}
        row.update({label: value_after_label(text, label) for label in labels})
        if row.get("Machine") and row.get("Process start"):
            row["Extruder"] = extruder_number(row["Machine"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["Process start"] = frame["Process start"].map(parse_dt)
    frame["Process end"] = frame["Process end"].map(parse_dt)
    return (frame.drop_duplicates(["Stage", "Machine", "Process start", "Process end"])
            .sort_values("Source Page").reset_index(drop=True))


def parse_recipe_settings(pages, summaries):
    rows = []
    pattern = re.compile(
        r"^\s*(.*?)\s{2,}(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(\S+)\s*$"
    )
    for _, summary in summaries.iterrows():
        stage = summary["Stage"]
        first = int(summary["Source Page"]) - 1
        active = False
        for page_index in range(first, min(first + 5, len(pages))):
            text = pages[page_index]
            if re.search(rf"(?mi)^\s*{stage}\s+RECIPE SETTINGS\s*$", text):
                active = True
            if not active:
                continue
            for line in text.splitlines():
                if "RECIPE SETTINGS" in line.upper():
                    continue
                match = pattern.match(line)
                if not match:
                    continue
                parameter = match.group(1).strip()
                if not parameter or parameter.lower() == "name":
                    continue
                rows.append({
                    "Stage": stage, "Extruder": summary["Extruder"],
                    "Machine": summary["Machine"], "Parameter": parameter,
                    "Minimum": float(match.group(2)), "Set Value": float(match.group(3)),
                    "Maximum": float(match.group(4)), "Unit": match.group(5),
                    "Source Page": page_index + 1,
                })
            if re.search(rf"(?mi)^\s*{stage}\s+DATALOG\s*$", text):
                break
    columns = ["Stage", "Extruder", "Machine", "Parameter", "Minimum",
               "Set Value", "Maximum", "Unit", "Source Page"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (pd.DataFrame(rows).drop_duplicates(
        ["Stage", "Extruder", "Machine", "Parameter", "Source Page"]
    ).reset_index(drop=True))


def summary_for_datalog_page(stage, page_number, summaries):
    """Assign by report block/page order, not time.

    Multiple extruders run concurrently, so timestamp matching can assign a page
    to the wrong extruder. A datalog belongs to the latest preceding summary page
    for the same stage.
    """
    if summaries.empty:
        return None
    candidates = summaries[
        summaries["Stage"].eq(stage)
        & (summaries["Source Page"] < page_number)
    ].sort_values("Source Page")
    return candidates.iloc[-1] if not candidates.empty else None


def parse_datalogs(pages, summaries):
    """Extract each process block from its own contiguous datalog pages.

    The parser starts at the relevant process-summary page, waits until it finds
    that stage's first DATALOG page, and then continues until the first page that
    no longer carries the same datalog heading. This prevents data from the next
    extruder or a later report section from being joined to the selected run.
    """
    extracted_frames = []
    diagnostic_rows = []
    rejected_rows = []
    process_ranges = []
    timestamp_start = re.compile(rf"^\s*({DATE})\s+({TIME})\s+")

    ordered = summaries.sort_values("Source Page").reset_index(drop=True)

    for _, summary in ordered.iterrows():
        stage = summary["Stage"]
        machine = summary["Machine"]
        extruder = summary["Extruder"]
        summary_page = int(summary["Source Page"])
        expected_fields = len(DATALOG_COLUMNS[stage])
        row_pattern = re.compile(
            rf"^\s*({DATE})\s+({TIME})\s+(.+?)\s+"
            rf"((?:\S+\s+){{{expected_fields - 1}}}\S+)\s*$"
        )

        started = False
        first_datalog_page = None
        last_datalog_page = None
        run_frames = []
        terminal_step_found = False
        terminal_step_name = None

        for page_number in range(summary_page, len(pages) + 1):
            text = pages[page_number - 1]
            heading_found = bool(
                re.search(rf"(?mi)^\s*{stage}\s+DATALOG\s*$", text)
            )

            if not started:
                if not heading_found:
                    # Recipe settings may occupy a few pages after the summary.
                    # Stop if a different target process summary is reached.
                    different_summary = identify_stage(text, "PROCESS")
                    if page_number > summary_page and different_summary:
                        break
                    continue
                started = True
                first_datalog_page = page_number

            elif not heading_found:
                # The contiguous datalog block for this run has ended.
                break

            last_datalog_page = page_number
            records = []
            candidate_count = 0
            rejected_count = 0

            for line_number, line in enumerate(text.splitlines(), 1):
                if not timestamp_start.match(line):
                    continue
                candidate_count += 1
                match = row_pattern.match(line)
                if not match:
                    rejected_count += 1
                    rejected_rows.append({
                        "Stage": stage, "Machine": machine, "Extruder": extruder,
                        "Source Page": page_number, "Line Number": line_number,
                        "Reason": f"Could not isolate {expected_fields} trailing fields",
                        "Raw Line": line.strip(),
                    })
                    continue

                raw_values = match.group(4).split()
                if len(raw_values) != expected_fields:
                    rejected_count += 1
                    rejected_rows.append({
                        "Stage": stage, "Machine": machine, "Extruder": extruder,
                        "Source Page": page_number, "Line Number": line_number,
                        "Reason": f"Expected {expected_fields} fields; found {len(raw_values)}",
                        "Raw Line": line.strip(),
                    })
                    continue

                converted = []
                for value in raw_values:
                    if value.upper() == "NA":
                        converted.append(None)
                    else:
                        try:
                            converted.append(float(value))
                        except ValueError:
                            converted.append(value)

                record = {
                    "Stage": stage,
                    "Machine": machine,
                    "Extruder": extruder,
                    "Date & Time": parse_dt(f"{match.group(1)} {match.group(2)}"),
                    "Step Name": match.group(3).strip(),
                    "Source Page": page_number,
                }
                record.update(dict(zip(DATALOG_COLUMNS[stage], converted)))
                records.append(record)

                # Blending is complete when the report records "Blending is done".
                # Include the terminal row, then stop the blending block at this page.
                normalized_step = re.sub(r"\s+", " ", record["Step Name"]).strip().casefold()
                if stage == "BLENDING" and normalized_step in {
                    "blending is done",
                    "blending is completed",
                    "blending completed",
                }:
                    terminal_step_found = True
                    terminal_step_name = record["Step Name"]
                    break

            diagnostic_rows.append({
                "Stage": stage, "Machine": machine, "Extruder": extruder,
                "Summary Page": summary_page, "Source Page": page_number,
                "Expected Fields": expected_fields,
                "Candidate Timestamp Rows": candidate_count,
                "Extracted Rows": len(records),
                "Rejected/Missing Rows": rejected_count,
                "Extraction Rate (%)": (
                    round(100 * len(records) / candidate_count, 2)
                    if candidate_count else 100.0
                ),
                "Status": "OK" if rejected_count == 0 else "REVIEW",
            })
            if records:
                run_frames.append(pd.DataFrame(records))

            if terminal_step_found:
                break

        process_ranges.append({
            "Stage": stage, "Machine": machine, "Extruder": extruder,
            "Summary Page": summary_page,
            "First Datalog Page": first_datalog_page,
            "Last Datalog Page": last_datalog_page,
            "Datalog Pages": (
                last_datalog_page - first_datalog_page + 1
                if first_datalog_page is not None and last_datalog_page is not None
                else 0
            ),
            "Terminal Step Found": terminal_step_found,
            "Terminal Step Name": terminal_step_name,
            "Status": (
                "OK"
                if first_datalog_page is not None
                and (stage != "BLENDING" or terminal_step_found)
                else "DATALOG NOT FOUND"
                if first_datalog_page is None
                else "TERMINAL STEP NOT FOUND"
            ),
        })
        if run_frames:
            extracted_frames.append(pd.concat(run_frames, ignore_index=True))

    datalog = (
        pd.concat(extracted_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["Stage", "Extruder", "Date & Time", "Source Page"])
        .reset_index(drop=True)
        if extracted_frames else pd.DataFrame()
    )
    if not datalog.empty:
        preferred = ["Stage", "Machine", "Extruder", "Date & Time", "Step Name", "Source Page"]
        remaining = [column for column in datalog.columns if column not in preferred]
        datalog = datalog[[column for column in preferred if column in datalog.columns] + remaining]

    return (
        datalog,
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(rejected_rows),
        pd.DataFrame(process_ranges),
    )


def make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected, process_ranges):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([batch]).to_excel(writer, "Batch Details", index=False)
        overview.to_excel(writer, "Process Overview", index=False)
        summaries.to_excel(writer, "Process Summary", index=False)
        settings.to_excel(writer, "Recipe Settings", index=False)
        process_ranges.to_excel(writer, "Process Page Ranges", index=False)
        diagnostics.to_excel(writer, "Datalog Diagnostics", index=False)
        rejected.to_excel(writer, "Rejected Rows", index=False)
        for stage in STAGES:
            part = datalog[datalog["Stage"].eq(stage)] if not datalog.empty else pd.DataFrame()
            part.to_excel(writer, f"{stage.title()} Datalog"[:31], index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.sheet_view.showGridLines = False
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for column in worksheet.columns:
                letter = column[0].column_letter
                worksheet.column_dimensions[letter].width = min(
                    max(len(str(cell.value or "")) for cell in column) + 2, 40
                )
    return output.getvalue()


def main():
    st.title("🧪 Complete Batch Process Report Extractor")
    st.caption("Extracts Process Overview, recipe settings, and complete Wetting, Premixing, Mixing and Blending datalogs.")
    uploaded = st.file_uploader("Upload Batch Cycle Report PDF", type=["pdf"])
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
            datalog, diagnostics, rejected_rows, process_ranges = parse_datalogs(pages, summaries)
            status.update(label="Extraction completed", state="complete", expanded=False)
    except Exception as error:
        st.exception(error)
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF pages", f"{len(pages):,}")
    c2.metric("Overview operations", f"{len(overview):,}")
    c3.metric("Target process runs", f"{len(summaries):,}")
    c4.metric("Datalog rows", f"{len(datalog):,}")

    st.subheader("Batch Details")
    st.dataframe(pd.DataFrame([batch]), use_container_width=True, hide_index=True)
    st.subheader("Process Overview Times")
    st.dataframe(overview, use_container_width=True, hide_index=True)
    st.subheader("Process Summary")
    st.dataframe(summaries, use_container_width=True, hide_index=True)

    recipe_tab, datalog_tab, diagnostics_tab = st.tabs([
        "Recipe Settings", "Datalog Explorer", "Diagnostics"
    ])

    with recipe_tab:
        st.dataframe(settings, use_container_width=True, hide_index=True, height=520)
        st.download_button("Download Recipe Settings CSV", settings.to_csv(index=False).encode("utf-8"),
                           "recipe_settings.csv", "text/csv")

    with datalog_tab:
        if datalog.empty:
            st.warning("No datalog rows were extracted.")
        else:
            stage_options = [stage for stage in STAGES if stage in set(datalog["Stage"])]
            stage = st.selectbox("Stage", stage_options)
            stage_data = datalog[datalog["Stage"].eq(stage)]
            extruders = sorted(int(value) for value in stage_data["Extruder"].dropna().unique())
            extruder = st.selectbox("Extruder", extruders) if extruders else None
            view = stage_data[stage_data["Extruder"].eq(extruder)] if extruder is not None else stage_data
            steps = sorted(view["Step Name"].dropna().unique())
            chosen_steps = st.multiselect("Step Name", steps, default=steps)
            view = view[view["Step Name"].isin(chosen_steps)]
            pages_found = sorted(int(value) for value in view["Source Page"].dropna().unique())
            st.write(f"Rows: **{len(view):,}** | PDF pages: **{pages_found[0] if pages_found else '-'} to {pages_found[-1] if pages_found else '-'}**")
            st.dataframe(view, use_container_width=True, hide_index=True, height=560)
            st.download_button("Download Filtered Datalog CSV", view.to_csv(index=False).encode("utf-8"),
                               f"{stage.lower()}_extruder_{extruder}.csv", "text/csv")

    with diagnostics_tab:
        st.markdown("#### Assigned process page ranges")
        if process_ranges.empty:
            st.warning("No process page ranges could be constructed.")
        else:
            st.dataframe(process_ranges, use_container_width=True, hide_index=True)

        if diagnostics.empty:
            st.warning("No datalog pages were detected.")
        else:
            stage_audit = (diagnostics.groupby(["Stage", "Machine", "Extruder"], dropna=False, as_index=False)
                           .agg(**{
                               "First Page": ("Source Page", "min"),
                               "Last Page": ("Source Page", "max"),
                               "Datalog Pages": ("Source Page", "nunique"),
                               "Candidate Rows": ("Candidate Timestamp Rows", "sum"),
                               "Extracted Rows": ("Extracted Rows", "sum"),
                               "Rejected Rows": ("Rejected/Missing Rows", "sum"),
                           }))
            stage_audit["Extraction Rate (%)"] = (
                100 * stage_audit["Extracted Rows"] / stage_audit["Candidate Rows"].replace(0, pd.NA)
            ).round(2).fillna(100.0)
            stage_audit["Status"] = stage_audit["Rejected Rows"].map(lambda x: "OK" if x == 0 else "REVIEW")
            st.markdown("#### Stage and extruder coverage")
            st.dataframe(stage_audit, use_container_width=True, hide_index=True)
            st.markdown("#### Page-level diagnostics")
            review_only = st.checkbox("Show only pages requiring review", value=False)
            page_view = diagnostics[diagnostics["Status"].eq("REVIEW")] if review_only else diagnostics
            st.dataframe(page_view, use_container_width=True, hide_index=True, height=400)

        st.markdown("#### Rejected rows")
        if rejected_rows.empty:
            st.success("No timestamp rows were rejected.")
        else:
            st.warning(f"{len(rejected_rows):,} rows require review.")
            st.dataframe(rejected_rows, use_container_width=True, hide_index=True, height=400)
            st.download_button("Download Rejected Rows CSV", rejected_rows.to_csv(index=False).encode("utf-8"),
                               "rejected_datalog_rows.csv", "text/csv")

    excel = make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected_rows, process_ranges)
    batch_number = batch.get("Batch no.") or "batch"
    st.download_button("⬇️ Download Complete Excel Extraction", excel,
                       f"{batch_number}_complete_process_extraction.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")


if __name__ == "__main__":
    main()
