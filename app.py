import io
import re

import matplotlib.pyplot as plt
import numpy as np
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

# Premixing engineering constants
PREMIXING_VOLUME_STEP_ML = 16.5
PREMIXING_PRESSURE_STEP_BAR = 15.0
PREMIXING_REPEAT_CYCLES = 3
PREMIXING_MAX_PRESSURE_BAR = 190.0
PREMIXING_CYLINDER_DIAMETER_MM = 80.1
PREMIXING_CYLINDER_AREA_MM2 = (
    np.pi * PREMIXING_CYLINDER_DIAMETER_MM ** 2 / 4
)
PREMIXING_VOLUME_PER_MM_ML = PREMIXING_CYLINDER_AREA_MM2 / 1000
PREMIXING_TARGET_ACHIEVEMENT_PERCENT = 95.0
PREMIXING_PRESSURE_MATCH_TOLERANCE_BAR = 2.0
PREMIXING_SPEED_DIRECTION_TOLERANCE = 0.005


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
    parser_version = "2026.10.13-premixing-strict-pressure-increase-v7"
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


def normalize_mixing_step_name(step_name):
    """Correct report typo 3,8 mm/3,8mm to 3.8mm in Mixing Step Name."""
    text = re.sub(r"(?i)3\s*,\s*8\s*mm", "3.8mm", str(step_name))
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
                raw_step_name = match.group(3).strip()
                corrected_step_name = (
                    normalize_mixing_step_name(raw_step_name)
                    if stage == "MIXING" else raw_step_name
                )
                record = {"Stage": stage, "Machine": machine, "Extruder": extruder, "Date & Time": parse_dt(f"{match.group(1)} {match.group(2)}"), "Step Name": corrected_step_name, "Source Page": page_number, "Source Line": line_number}
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



def premixing_movement_direction(left_speed, right_speed):
    """Infer direction from the two servo-speed signs."""
    if pd.isna(left_speed) or pd.isna(right_speed):
        return ""
    left_speed = float(left_speed)
    right_speed = float(right_speed)
    tolerance = PREMIXING_SPEED_DIRECTION_TOLERANCE
    if left_speed > tolerance and right_speed < -tolerance:
        return "L-to-R"
    if left_speed < -tolerance and right_speed > tolerance:
        return "R-to-L"
    return ""


def prepare_premixing_engineering_data(premixing_data):
    """Prepare Premixing datalog for movement-block analysis."""
    if premixing_data.empty:
        return pd.DataFrame()
    data = premixing_data.copy()
    numeric_columns = [
        "Current Cycle", "Loadcell Left (bar)", "Loadcell Right (bar)",
        "Servo Position Left (mm)", "Servo Position Right (mm)",
        "Servo Speed Left (mm/s)", "Servo Speed Right (mm/s)",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(
        ["Extruder", "Source Page", "Source Line"], kind="stable"
    ).reset_index(drop=True)
    data["Elapsed Time (min)"] = data.groupby("Extruder")["Date & Time"].transform(
        lambda values: (values - values.min()).dt.total_seconds() / 60
    )
    data["Direction"] = [
        premixing_movement_direction(left, right)
        for left, right in zip(
            data["Servo Speed Left (mm/s)"],
            data["Servo Speed Right (mm/s)"],
        )
    ]
    context = (
        data["Step Name"].fillna("").astype(str).str.casefold()
        .str.contains(r"premix|accumulator|void reduction", regex=True)
    )
    data.loc[~context, "Direction"] = ""

    all_ids = []
    for _, group in data.groupby("Extruder", sort=False):
        block_id = 0
        prior_direction = ""
        active = False
        for direction in group["Direction"]:
            if direction:
                if not active or direction != prior_direction:
                    block_id += 1
                all_ids.append(block_id)
                active = True
                prior_direction = direction
            else:
                all_ids.append(pd.NA)
                active = False
                prior_direction = ""
    data["Directional Block ID"] = pd.array(all_ids, dtype="Int64")
    return data


def summarize_premixing_directional_blocks(prepared_data):
    """Summarize every continuous directional Premixing movement."""
    columns = [
        "Extruder", "Directional Block ID", "Logged Cycle", "Direction",
        "Start Date & Time", "End Date & Time", "Duration (s)",
        "Start Source Page", "End Source Page", "Start Source Line", "End Source Line",
        "Left Start Position (mm)", "Left End Position (mm)",
        "Right Start Position (mm)", "Right End Position (mm)",
        "Left Movement (mm)", "Right Movement (mm)",
        "Left Volume (mL)", "Right Volume (mL)",
        "Pushed Volume (mL)", "Received Volume (mL)",
        "Volume Difference (mL)", "Average Actual Speed (mL/min)",
        "Maximum Push Pressure (bar)", "Maximum Receiving Pressure (bar)",
    ]
    if prepared_data.empty:
        return pd.DataFrame(columns=columns)

    output = []
    for extruder, extruder_data in prepared_data.groupby("Extruder", sort=True):
        moving = extruder_data.dropna(subset=["Directional Block ID"])
        for block_id, group in moving.groupby("Directional Block ID", sort=True):
            group = group.sort_values(["Source Page", "Source Line"], kind="stable")
            direction = group["Direction"].iloc[0]
            first_index = group.index[0]
            previous_index = extruder_data.index[extruder_data.index < first_index]
            start = (
                extruder_data.loc[previous_index[-1]]
                if len(previous_index) else group.iloc[0]
            )
            end = group.iloc[-1]
            left_delta = float(end["Servo Position Left (mm)"] - start["Servo Position Left (mm)"])
            right_delta = float(end["Servo Position Right (mm)"] - start["Servo Position Right (mm)"])
            left_volume = abs(left_delta) * PREMIXING_VOLUME_PER_MM_ML
            right_volume = abs(right_delta) * PREMIXING_VOLUME_PER_MM_ML
            if direction == "L-to-R":
                pushed, received = left_volume, right_volume
                max_push = group["Loadcell Left (bar)"].max()
                max_receive = group["Loadcell Right (bar)"].max()
            else:
                pushed, received = right_volume, left_volume
                max_push = group["Loadcell Right (bar)"].max()
                max_receive = group["Loadcell Left (bar)"].max()
            duration = (end["Date & Time"] - start["Date & Time"]).total_seconds()
            cycles = group["Current Cycle"].dropna()
            logged_cycle = int(cycles.mode().iloc[0]) if not cycles.empty else pd.NA
            output.append({
                "Extruder": int(extruder),
                "Directional Block ID": int(block_id),
                "Logged Cycle": logged_cycle,
                "Direction": direction,
                "Start Date & Time": start["Date & Time"],
                "End Date & Time": end["Date & Time"],
                "Duration (s)": duration,
                "Start Source Page": int(group["Source Page"].iloc[0]),
                "End Source Page": int(group["Source Page"].iloc[-1]),
                "Start Source Line": int(group["Source Line"].iloc[0]),
                "End Source Line": int(group["Source Line"].iloc[-1]),
                "Left Start Position (mm)": start["Servo Position Left (mm)"],
                "Left End Position (mm)": end["Servo Position Left (mm)"],
                "Right Start Position (mm)": start["Servo Position Right (mm)"],
                "Right End Position (mm)": end["Servo Position Right (mm)"],
                "Left Movement (mm)": left_delta,
                "Right Movement (mm)": right_delta,
                "Left Volume (mL)": left_volume,
                "Right Volume (mL)": right_volume,
                "Pushed Volume (mL)": pushed,
                "Received Volume (mL)": received,
                "Volume Difference (mL)": abs(pushed - received),
                "Average Actual Speed (mL/min)": (
                    pushed / duration * 60 if duration > 0 else np.nan
                ),
                "Maximum Push Pressure (bar)": max_push,
                "Maximum Receiving Pressure (bar)": max_receive,
            })
    return pd.DataFrame(output, columns=columns)


def pair_premixing_bidirectional_attempts(directional):
    """Pair adjacent opposite-direction blocks with cycle-aware scoring.

    Either L-to-R/R-to-L or R-to-L/L-to-R order is accepted. A same-cycle
    opposite-direction block is preferred; otherwise the nearest immediately
    following opposite-direction block is used and marked for review.
    """
    columns = [
        "Extruder", "Bidirectional Attempt ID", "Pairing Status",
        "Start Date & Time", "End Date & Time",
        "First Direction", "Second Direction", "First Block ID", "Second Block ID",
        "Logged Cycle First", "Logged Cycle Second", "Actual Logged Cycle",
        "L-to-R Pushed Volume (mL)", "R-to-L Pushed Volume (mL)",
        "Representative Achieved Volume (mL)",
        "L-to-R Maximum Push Pressure (bar)",
        "R-to-L Maximum Push Pressure (bar)",
        "Maximum Actual Pressure (bar)",
        "L-to-R Average Speed (mL/min)", "R-to-L Average Speed (mL/min)",
    ]
    if directional.empty:
        return pd.DataFrame(columns=columns)
    output = []
    for extruder, group in directional.groupby("Extruder", sort=True):
        data = group.sort_values("Start Date & Time").reset_index(drop=True)
        used = set()
        attempt_id = 0
        for i in range(len(data)):
            if i in used:
                continue
            first = data.iloc[i]
            candidates = []
            for j in range(i + 1, min(i + 4, len(data))):
                if j in used or data.iloc[j]["Direction"] == first["Direction"]:
                    continue
                second = data.iloc[j]
                same_cycle = (
                    pd.notna(first["Logged Cycle"])
                    and pd.notna(second["Logged Cycle"])
                    and int(first["Logged Cycle"]) == int(second["Logged Cycle"])
                )
                candidates.append((0 if same_cycle else 1, j, second))
            if not candidates:
                continue
            _, j, second = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
            used.update({i, j})
            attempt_id += 1
            blocks = {first["Direction"]: first, second["Direction"]: second}
            ltr = blocks.get("L-to-R")
            rtl = blocks.get("R-to-L")
            same_cycle = (
                pd.notna(first["Logged Cycle"])
                and pd.notna(second["Logged Cycle"])
                and int(first["Logged Cycle"]) == int(second["Logged Cycle"])
            )
            actual_cycle = (
                int(first["Logged Cycle"])
                if same_cycle else
                int(first["Logged Cycle"]) if pd.notna(first["Logged Cycle"])
                else int(second["Logged Cycle"]) if pd.notna(second["Logged Cycle"])
                else pd.NA
            )
            output.append({
                "Extruder": int(extruder),
                "Bidirectional Attempt ID": attempt_id,
                "Pairing Status": "Same logged cycle" if same_cycle else "Nearest opposite block - review",
                "Start Date & Time": min(first["Start Date & Time"], second["Start Date & Time"]),
                "End Date & Time": max(first["End Date & Time"], second["End Date & Time"]),
                "First Direction": first["Direction"],
                "Second Direction": second["Direction"],
                "First Block ID": int(first["Directional Block ID"]),
                "Second Block ID": int(second["Directional Block ID"]),
                "Logged Cycle First": first["Logged Cycle"],
                "Logged Cycle Second": second["Logged Cycle"],
                "Actual Logged Cycle": actual_cycle,
                "L-to-R Pushed Volume (mL)": ltr["Pushed Volume (mL)"],
                "R-to-L Pushed Volume (mL)": rtl["Pushed Volume (mL)"],
                "Representative Achieved Volume (mL)": min(
                    ltr["Pushed Volume (mL)"], rtl["Pushed Volume (mL)"]
                ),
                "L-to-R Maximum Push Pressure (bar)": ltr["Maximum Push Pressure (bar)"],
                "R-to-L Maximum Push Pressure (bar)": rtl["Maximum Push Pressure (bar)"],
                "Maximum Actual Pressure (bar)": max(
                    ltr["Maximum Push Pressure (bar)"], rtl["Maximum Push Pressure (bar)"]
                ),
                "L-to-R Average Speed (mL/min)": ltr["Average Actual Speed (mL/min)"],
                "R-to-L Average Speed (mL/min)": rtl["Average Actual Speed (mL/min)"],
            })
    return pd.DataFrame(output, columns=columns)


def predict_premixing_pressure_level(actual_pressure):
    if pd.isna(actual_pressure):
        return np.nan, np.nan
    actual_pressure = float(actual_pressure)
    if actual_pressure <= 0:
        return 0, 0.0
    step_number = max(
        1,
        int(np.floor((actual_pressure + PREMIXING_PRESSURE_STEP_BAR / 2)
                     / PREMIXING_PRESSURE_STEP_BAR)),
    )
    target = min(
        step_number * PREMIXING_PRESSURE_STEP_BAR,
        PREMIXING_MAX_PRESSURE_BAR,
    )
    return step_number, target


def reverse_predict_premixing_volume_step(paired_attempts):
    """Reconstruct Premixing volume step from cycle and actual pressure increases.

    Minimal process rule:
    - Start at 16.5 mL.
    - Logged cycle 2 or 3 means another attempt at the same volume step.
    - When the cycle returns to 1, compare the current maximum actual pressure
      with the highest maximum actual pressure already observed in that extruder.
    - Any positive increase, even 0.1 bar, is a pressure increase and keeps the
      same volume step.
    - If cycle returns to 1 without a pressure increase, advance the cumulative
      volume step by 16.5 mL.

    No pressure rounding, 15-bar binning, or pressure tolerance is used for
    deciding whether the volume step advances.
    """
    if paired_attempts.empty:
        return pd.DataFrame()

    output = []
    for extruder, attempts in paired_attempts.groupby("Extruder", sort=True):
        volume_step_number = 1
        volume_step_size = PREMIXING_VOLUME_STEP_ML
        highest_pressure_seen = None

        for sequence, (_, row) in enumerate(
            attempts.sort_values("Start Date & Time").iterrows(), start=1
        ):
            actual_cycle = row["Actual Logged Cycle"]
            actual_pressure = float(row["Maximum Actual Pressure (bar)"])

            pressure_increased = (
                highest_pressure_seen is not None
                and actual_pressure > highest_pressure_seen
            )
            previous_highest_pressure = highest_pressure_seen

            if sequence == 1:
                rule = "Initial volume step"
            elif pd.notna(actual_cycle) and int(actual_cycle) == 1:
                if pressure_increased:
                    rule = "Same volume step: pressure increased"
                else:
                    volume_step_number += 1
                    volume_step_size = (
                        volume_step_number * PREMIXING_VOLUME_STEP_ML
                    )
                    rule = "Advanced by 16.5 mL"
            else:
                rule = "Same volume step: repeat attempt"

            highest_pressure_seen = (
                actual_pressure
                if highest_pressure_seen is None
                else max(highest_pressure_seen, actual_pressure)
            )

            output.append({
                **row.to_dict(),
                "Attempt Number": sequence,
                "Volume step size (mL)": volume_step_size,
                "Volume Step Number": volume_step_number,
                "Volume Step Rule": rule,
                "Previous Highest Pressure (bar)": previous_highest_pressure,
                "Highest Pressure Seen (bar)": highest_pressure_seen,
                "Pressure Increased": "Yes" if pressure_increased else "No",
            })

    return pd.DataFrame(output)


def build_premixing_volume_step_summary(prediction):
    """Summarize attempts and directional maximum pressures by volume step."""
    columns = [
        "Extruder", "Volume Step (mL)", "Attempts",
        "Max Loadcell Left (bar)", "Max Loadcell Right (bar)",
        "First Attempt", "Last Attempt",
    ]
    if prediction.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (extruder, volume_step), group in prediction.groupby(
        ["Extruder", "Volume step size (mL)"], sort=True
    ):
        rows.append({
            "Extruder": int(extruder),
            "Volume Step (mL)": float(volume_step),
            "Attempts": len(group),
            "Max Loadcell Left (bar)": group[
                "L-to-R Maximum Push Pressure (bar)"
            ].max(),
            "Max Loadcell Right (bar)": group[
                "R-to-L Maximum Push Pressure (bar)"
            ].max(),
            "First Attempt": group["Start Date & Time"].min(),
            "Last Attempt": group["End Date & Time"].max(),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Extruder", "Volume Step (mL)"]
    ).reset_index(drop=True)


def create_premixing_volume_step_plot(summary, extruder, batch_number):
    """Create an interactive dual-axis Premixing pressure/attempt plot."""
    data = summary[summary["Extruder"].eq(extruder)].sort_values(
        "Volume Step (mL)"
    ).copy()
    figure = go.Figure()

    date_details = data[["First Attempt", "Last Attempt"]].copy()
    for column in date_details.columns:
        date_details[column] = pd.to_datetime(
            date_details[column], errors="coerce"
        ).dt.strftime("%d-%m-%Y %H:%M:%S")

    figure.add_trace(go.Bar(
        x=data["Volume Step (mL)"],
        y=data["Attempts"],
        name="Attempts",
        yaxis="y2",
        width=PREMIXING_VOLUME_STEP_ML * 0.62,
        marker={
            "color": "rgba(181, 211, 143, 0.55)",
            "line": {"color": "rgba(123, 158, 83, 0.85)", "width": 1},
        },
        customdata=date_details.to_numpy(),
        hovertemplate=(
            "<b>Attempts</b><br>Volume step: %{x:.1f} mL<br>"
            "Number of attempts: %{y:.0f}<br>"
            "First attempt: %{customdata[0]}<br>"
            "Last attempt: %{customdata[1]}<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter(
        x=data["Volume Step (mL)"],
        y=data["Max Loadcell Left (bar)"],
        mode="lines+markers",
        name="Max Loadcell Left (bar)",
        line={"color": "#0057B8", "width": 4, "shape": "hv"},
        marker={"size": 9, "color": "#0057B8", "symbol": "circle"},
        hovertemplate=(
            "<b>Max Loadcell Left</b><br>Volume step: %{x:.1f} mL<br>"
            "Maximum pressure: %{y:.2f} bar<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter(
        x=data["Volume Step (mL)"],
        y=data["Max Loadcell Right (bar)"],
        mode="lines+markers",
        name="Max Loadcell Right (bar)",
        line={"color": "#D62728", "width": 4, "shape": "hv"},
        marker={"size": 9, "color": "#D62728", "symbol": "diamond"},
        hovertemplate=(
            "<b>Max Loadcell Right</b><br>Volume step: %{x:.1f} mL<br>"
            "Maximum pressure: %{y:.2f} bar<extra></extra>"
        ),
    ))

    x_max = max(
        231.0,
        float(data["Volume Step (mL)"].max()) + PREMIXING_VOLUME_STEP_ML
        if not data.empty else 231.0,
    )
    max_attempts = int(data["Attempts"].max()) if not data.empty else 1
    figure.update_layout(
        title={
            "text": f"{batch_number} | EXTRUDER-{int(extruder)} PRE-MIXING",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 24, "color": "#1F2937"},
        },
        xaxis={
            "title": {"text": "VOLUME STEP (mL)", "standoff": 18},
            "range": [0, x_max],
            "dtick": PREMIXING_VOLUME_STEP_ML,
            "showgrid": True,
            "gridcolor": "#D9E1EA",
        },
        yaxis={
            "title": {"text": "MAX PRESSURE (bar)", "standoff": 12},
            "range": [0, 200],
            "dtick": 20,
            "showgrid": True,
            "gridcolor": "#D9E1EA",
        },
        yaxis2={
            "title": {"text": "ATTEMPTS", "standoff": 12},
            "overlaying": "y",
            "side": "right",
            "range": [0, max(4, max_attempts + 1)],
            "dtick": 1,
            "showgrid": False,
        },
        barmode="overlay",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.03,
            "yanchor": "bottom",
            "font": {"size": 13},
            "bgcolor": "rgba(255,255,255,0.9)",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=720,
        margin={"l": 90, "r": 90, "t": 150, "b": 90},
    )
    return figure


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



def build_mixing_max_pressure_table(mixing_data):
    """Calculate directional maximum pressure for each extrusion cycle.

    Physical direction mapping:
    - Movement to accumulator RIGHT: maximum Loadcell LEFT pressure.
    - Movement to accumulator LEFT: maximum Loadcell RIGHT pressure.

    Separate results are generated for 5 mm, 3.8 mm and 2 mm extrusion steps.
    Cycle 0 is excluded because it represents initialization rather than a completed cycle.
    """
    columns = [
        "Extruder", "Orifice", "Cycle",
        "Max Pressure Left - Move Right (bar)",
        "Max Pressure Right - Move Left (bar)",
        "Move Right Start", "Move Right End",
        "Move Left Start", "Move Left End",
        "Move Right Source Pages", "Move Left Source Pages",
    ]
    if mixing_data.empty:
        return pd.DataFrame(columns=columns)

    work = mixing_data.copy()
    work["Current Cycle"] = pd.to_numeric(work["Current Cycle"], errors="coerce")
    work["Loadcell Left (bar)"] = pd.to_numeric(work["Loadcell Left (bar)"], errors="coerce")
    work["Loadcell Right (bar)"] = pd.to_numeric(work["Loadcell Right (bar)"], errors="coerce")
    work["Step Name"] = work["Step Name"].map(normalize_mixing_step_name)
    work["Normalized Step"] = (
        work["Step Name"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )

    patterns = {
        "5 mm": r"extrusion\s*:?\s*5\s*mm\s+to\s+accumulator\s+(?:right|left)",
        "3.8 mm": r"extrusion\s*:?\s*3[\.,]8\s*mm\s+to\s+accumulator\s+(?:right|left)",
        "2 mm": r"extrusion\s*:?\s*2\s*mm\s+to\s+accumulator\s+(?:right|left)",
    }
    output = []

    for extruder, extruder_data in work.groupby("Extruder", sort=True):
        for orifice, pattern in patterns.items():
            stage_data = extruder_data[
                extruder_data["Normalized Step"].str.contains(pattern, regex=True, na=False)
            ].copy()
            stage_data = stage_data[stage_data["Current Cycle"].gt(0)]
            if stage_data.empty:
                continue

            for cycle, cycle_data in stage_data.groupby("Current Cycle", sort=True):
                right_move = cycle_data[
                    cycle_data["Normalized Step"].str.endswith("accumulator right")
                ]
                left_move = cycle_data[
                    cycle_data["Normalized Step"].str.endswith("accumulator left")
                ]

                output.append({
                    "Extruder": int(extruder),
                    "Orifice": orifice,
                    "Cycle": int(cycle),
                    "Max Pressure Left - Move Right (bar)": (
                        right_move["Loadcell Left (bar)"].max()
                        if not right_move.empty else pd.NA
                    ),
                    "Max Pressure Right - Move Left (bar)": (
                        left_move["Loadcell Right (bar)"].max()
                        if not left_move.empty else pd.NA
                    ),
                    "Move Right Start": right_move["Date & Time"].min() if not right_move.empty else pd.NaT,
                    "Move Right End": right_move["Date & Time"].max() if not right_move.empty else pd.NaT,
                    "Move Left Start": left_move["Date & Time"].min() if not left_move.empty else pd.NaT,
                    "Move Left End": left_move["Date & Time"].max() if not left_move.empty else pd.NaT,
                    "Move Right Source Pages": (
                        ", ".join(map(str, sorted(right_move["Source Page"].dropna().astype(int).unique())))
                        if not right_move.empty else ""
                    ),
                    "Move Left Source Pages": (
                        ", ".join(map(str, sorted(left_move["Source Page"].dropna().astype(int).unique())))
                        if not left_move.empty else ""
                    ),
                })

    result = pd.DataFrame(output, columns=columns)
    if result.empty:
        return result
    pressure_columns = [
        "Max Pressure Left - Move Right (bar)",
        "Max Pressure Right - Move Left (bar)",
    ]
    result[pressure_columns] = result[pressure_columns].apply(pd.to_numeric, errors="coerce")
    return result.sort_values(["Extruder", "Orifice", "Cycle"]).reset_index(drop=True)



def build_blending_max_pressure_table(blending_data):
    """Calculate directional maximum pressure for Pre- and Post-pH blending cycles.

    Pre-pH blending comprises 60 cycles. Post-pH blending begins when the
    Current Cycle resets after the completed 60th Pre-pH cycle and comprises
    up to 100 cycles.

    Directional pressure mapping:
    - Blending to accumulator RIGHT: maximum Loadcell LEFT pressure.
    - Blending to accumulator LEFT: maximum Loadcell RIGHT pressure.
    """
    columns = [
        "Extruder", "Blending Phase", "Cycle",
        "Max Pressure Left - Move Right (bar)",
        "Max Pressure Right - Move Left (bar)",
        "Move Right Start", "Move Right End",
        "Move Left Start", "Move Left End",
        "Move Right Source Pages", "Move Left Source Pages",
    ]
    if blending_data.empty:
        return pd.DataFrame(columns=columns)

    work = blending_data.copy()
    work["Current Cycle"] = pd.to_numeric(work["Current Cycle"], errors="coerce")
    work["Loadcell Left (bar)"] = pd.to_numeric(work["Loadcell Left (bar)"], errors="coerce")
    work["Loadcell Right (bar)"] = pd.to_numeric(work["Loadcell Right (bar)"], errors="coerce")
    work["Normalized Step"] = (
        work["Step Name"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    work = work[
        work["Normalized Step"].str.match(
            r"^blending to accumulator (?:right|left)$", na=False
        )
        & work["Current Cycle"].gt(0)
    ].sort_values(["Extruder", "Source Page", "Source Line"], kind="stable")

    phase_frames = []
    for extruder, group in work.groupby("Extruder", sort=True):
        group = group.copy()
        phase = "Pre-pH Blending"
        previous_cycle = None
        phases = []
        for cycle in group["Current Cycle"]:
            if (
                previous_cycle is not None
                and previous_cycle >= 60
                and cycle <= 5
            ):
                phase = "Post-pH Blending"
            phases.append(phase)
            previous_cycle = cycle
        group["Blending Phase"] = phases
        phase_frames.append(group)

    if not phase_frames:
        return pd.DataFrame(columns=columns)
    work = pd.concat(phase_frames, ignore_index=True)

    output = []
    for (extruder, phase, cycle), cycle_data in work.groupby(
        ["Extruder", "Blending Phase", "Current Cycle"], sort=True
    ):
        right_move = cycle_data[
            cycle_data["Normalized Step"].eq("blending to accumulator right")
        ]
        left_move = cycle_data[
            cycle_data["Normalized Step"].eq("blending to accumulator left")
        ]
        output.append({
            "Extruder": int(extruder),
            "Blending Phase": phase,
            "Cycle": int(cycle),
            "Max Pressure Left - Move Right (bar)": (
                right_move["Loadcell Left (bar)"].max()
                if not right_move.empty else pd.NA
            ),
            "Max Pressure Right - Move Left (bar)": (
                left_move["Loadcell Right (bar)"].max()
                if not left_move.empty else pd.NA
            ),
            "Move Right Start": right_move["Date & Time"].min() if not right_move.empty else pd.NaT,
            "Move Right End": right_move["Date & Time"].max() if not right_move.empty else pd.NaT,
            "Move Left Start": left_move["Date & Time"].min() if not left_move.empty else pd.NaT,
            "Move Left End": left_move["Date & Time"].max() if not left_move.empty else pd.NaT,
            "Move Right Source Pages": (
                ", ".join(map(str, sorted(right_move["Source Page"].dropna().astype(int).unique())))
                if not right_move.empty else ""
            ),
            "Move Left Source Pages": (
                ", ".join(map(str, sorted(left_move["Source Page"].dropna().astype(int).unique())))
                if not left_move.empty else ""
            ),
        })

    result = pd.DataFrame(output, columns=columns)
    pressure_columns = [
        "Max Pressure Left - Move Right (bar)",
        "Max Pressure Right - Move Left (bar)",
    ]
    result[pressure_columns] = result[pressure_columns].apply(pd.to_numeric, errors="coerce")
    phase_order = pd.CategoricalDtype(
        ["Pre-pH Blending", "Post-pH Blending"], ordered=True
    )
    result["Blending Phase"] = result["Blending Phase"].astype(phase_order)
    return result.sort_values(
        ["Extruder", "Blending Phase", "Cycle"]
    ).reset_index(drop=True)


def create_blending_pressure_plot(table, extruder, phase, batch_number):
    """Create an interactive maximum-pressure plot for one Blending phase."""
    data = table[
        table["Extruder"].eq(extruder)
        & table["Blending Phase"].astype(str).eq(phase)
    ].sort_values("Cycle").copy()

    figure = go.Figure()
    series = [
        (
            "Max Pressure Left - Move Right (bar)",
            "Loadcell Left Max - movement to accumulator right",
            "#9BBB59", "triangle-up",
            ["Move Right Start", "Move Right End", "Move Right Source Pages"],
        ),
        (
            "Max Pressure Right - Move Left (bar)",
            "Loadcell Right Max - movement to accumulator left",
            "#4F81BD", "diamond",
            ["Move Left Start", "Move Left End", "Move Left Source Pages"],
        ),
    ]
    for column, name, color, symbol, detail_columns in series:
        values = data.dropna(subset=[column]).copy()
        if values.empty:
            continue
        custom = values[detail_columns].copy()
        for date_column in detail_columns[:2]:
            custom[date_column] = pd.to_datetime(
                custom[date_column], errors="coerce"
            ).dt.strftime("%d-%m-%Y %H:%M:%S")
        figure.add_trace(go.Scatter(
            x=values["Cycle"], y=values[column],
            mode="lines+markers", name=name,
            customdata=custom.to_numpy(),
            line={"color": color, "width": 2},
            marker={"color": color, "size": 8, "symbol": symbol},
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Phase: " + phase + "<br>"
                "Cycle: %{x}<br>"
                "Maximum pressure: %{y:.2f} bar<br>"
                "Movement start: %{customdata[0]}<br>"
                "Movement end: %{customdata[1]}<br>"
                "Source page(s): %{customdata[2]}"
                "<extra></extra>"
            ),
        ))

    expected_cycles = 60 if phase == "Pre-pH Blending" else 100
    max_pressure = data[[
        "Max Pressure Left - Move Right (bar)",
        "Max Pressure Right - Move Left (bar)",
    ]].max().max() if not data.empty else 160
    y_max = max(160, float(max_pressure) * 1.12) if pd.notna(max_pressure) else 160
    figure.update_layout(
        title={
            "text": f"{batch_number}<br>EXTRUDER - {int(extruder)} | {phase}",
            "x": 0.5, "xanchor": "center",
        },
        xaxis_title="Cycles", yaxis_title="Maximum Pressure (bar)",
        xaxis={"range": [0, expected_cycles + 2], "showgrid": True},
        yaxis={"range": [0, y_max], "showgrid": True},
        hovermode="closest",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.20},
        height=620,
        margin={"l": 70, "r": 30, "t": 100, "b": 120},
    )
    return figure

def create_mixing_pressure_plot(table, extruder, orifice, batch_number):
    """Create an interactive maximum-pressure-by-cycle chart."""
    data = table[
        table["Extruder"].eq(extruder) & table["Orifice"].eq(orifice)
    ].sort_values("Cycle").copy()

    figure = go.Figure()
    series = [
        (
            "Max Pressure Left - Move Right (bar)",
            "Loadcell Left Max - movement to accumulator right",
            "#4F81BD", "diamond",
            ["Move Right Start", "Move Right End", "Move Right Source Pages"],
        ),
        (
            "Max Pressure Right - Move Left (bar)",
            "Loadcell Right Max - movement to accumulator left",
            "#C0504D", "diamond",
            ["Move Left Start", "Move Left End", "Move Left Source Pages"],
        ),
    ]

    for column, name, color, symbol, detail_columns in series:
        values = data.dropna(subset=[column]).copy()
        if values.empty:
            continue
        custom = values[detail_columns].copy()
        for date_column in detail_columns[:2]:
            custom[date_column] = pd.to_datetime(custom[date_column], errors="coerce").dt.strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        figure.add_trace(go.Scatter(
            x=values["Cycle"],
            y=values[column],
            mode="lines+markers",
            name=name,
            customdata=custom.to_numpy(),
            line={"color": color, "width": 2, "dash": "dash" if "Right Max" in name else "solid"},
            marker={"color": color, "size": 9, "symbol": symbol},
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Cycle: %{x}<br>"
                "Maximum pressure: %{y:.2f} bar<br>"
                "Movement start: %{customdata[0]}<br>"
                "Movement end: %{customdata[1]}<br>"
                "Source page(s): %{customdata[2]}"
                "<extra></extra>"
            ),
        ))

    max_cycle = int(data["Cycle"].max()) if not data.empty else 1
    max_pressure = data[[
        "Max Pressure Left - Move Right (bar)",
        "Max Pressure Right - Move Left (bar)",
    ]].max().max() if not data.empty else 200
    y_max = max(200, float(max_pressure) * 1.12) if pd.notna(max_pressure) else 200

    figure.update_layout(
        title={
            "text": f"{batch_number}<br>EXTRUDER - {int(extruder)} | MIXING: Extrusion {orifice} cycles",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Cycles",
        yaxis_title="Maximum Pressure (bar)",
        xaxis={"dtick": 1, "range": [0.5, max_cycle + 0.5], "showgrid": True},
        yaxis={"range": [0, y_max], "showgrid": True},
        hovermode="closest",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.20},
        height=620,
        margin={"l": 70, "r": 30, "t": 100, "b": 120},
    )
    return figure

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


def make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected, process_ranges, wetting_rise_table, mixing_pressure_table, blending_pressure_table, premix_prepared, premix_blocks, premix_pairs, premix_prediction, premix_summary):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([batch]).to_excel(writer, "Batch Details", index=False)
        overview.to_excel(writer, "Process Overview", index=False); summaries.to_excel(writer, "Process Summary", index=False)
        settings.to_excel(writer, "Recipe Settings", index=False); process_ranges.to_excel(writer, "Process Page Ranges", index=False)
        wetting_rise_table.to_excel(writer, "Wetting Rise Time", index=False)
        mixing_pressure_table.to_excel(writer, "Mixing Max Pressure", index=False)
        blending_pressure_table.to_excel(writer, "Blending Max Pressure", index=False)
        premix_prepared.to_excel(writer, "Premix Calculated Datalog", index=False)
        premix_blocks.to_excel(writer, "Premix Direction Blocks", index=False)
        premix_pairs.to_excel(writer, "Premix Bidirectional", index=False)
        premix_prediction.to_excel(writer, "Premix Prediction", index=False)
        premix_summary.to_excel(writer, "Premix Volume Summary", index=False)
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
            mixing_data = datalog[datalog["Stage"].eq("MIXING")].copy() if not datalog.empty else pd.DataFrame()
            mixing_pressure_table = build_mixing_max_pressure_table(mixing_data)
            blending_data = datalog[datalog["Stage"].eq("BLENDING")].copy() if not datalog.empty else pd.DataFrame()
            blending_pressure_table = build_blending_max_pressure_table(blending_data)
            premixing_data = datalog[datalog["Stage"].eq("PREMIXING")].copy() if not datalog.empty else pd.DataFrame()
            premix_prepared = prepare_premixing_engineering_data(premixing_data)
            premix_blocks = summarize_premixing_directional_blocks(premix_prepared)
            premix_pairs = pair_premixing_bidirectional_attempts(premix_blocks)
            premix_prediction = reverse_predict_premixing_volume_step(premix_pairs)
            if not premix_pairs.empty and not premix_prediction.empty:
                volume_step_map = premix_prediction[[
                    "Extruder", "Bidirectional Attempt ID",
                    "Volume step size (mL)", "Volume Step Rule",
                ]].drop_duplicates([
                    "Extruder", "Bidirectional Attempt ID"
                ])
                premix_pairs = premix_pairs.merge(
                    volume_step_map,
                    on=["Extruder", "Bidirectional Attempt ID"],
                    how="left",
                )
            premix_summary = build_premixing_volume_step_summary(premix_prediction)
            status.update(label="Extraction completed", state="complete", expanded=False)
    except Exception as error: st.exception(error); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF pages", f"{len(pages):,}"); c2.metric("Overview operations", f"{len(overview):,}")
    c3.metric("Target process runs", f"{len(summaries):,}"); c4.metric("Datalog rows", f"{len(datalog):,}")
    st.subheader("Batch Details"); st.dataframe(pd.DataFrame([batch]), use_container_width=True, hide_index=True)
    st.subheader("Process Overview Times"); st.dataframe(overview, use_container_width=True, hide_index=True)
    st.subheader("Process Summary"); st.dataframe(summaries, use_container_width=True, hide_index=True)

    recipe_tab, datalog_tab, wetting_tab, mixing_tab, blending_tab, premixing_tab, diagnostics_tab = st.tabs(["Recipe Settings", "Datalog Explorer", "Wetting Rise-Time Plot", "Mixing Max-Pressure Plots", "Blending Max-Pressure Plots", "Premixing Engineering Analysis", "Diagnostics"])
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
    with mixing_tab:
        st.markdown("#### Mixing Maximum Pressure by Extrusion Cycle")
        st.caption(
            "For movement to accumulator right, the maximum left-loadcell pressure is used. "
            "For movement to accumulator left, the maximum right-loadcell pressure is used. "
            "Cycle 0 is excluded."
        )
        if mixing_pressure_table.empty:
            st.warning("No directional Mixing pressure-cycle data were generated.")
        else:
            mixing_extruders = sorted(mixing_pressure_table["Extruder"].unique().astype(int))
            selected_mixing_extruder = st.selectbox(
                "Mixing extruder", mixing_extruders, key="mixing_plot_extruder"
            )
            available_orifices = [
                item for item in ["5 mm", "3.8 mm", "2 mm"]
                if item in set(
                    mixing_pressure_table.loc[
                        mixing_pressure_table["Extruder"].eq(selected_mixing_extruder),
                        "Orifice",
                    ]
                )
            ]
            selected_orifices = st.multiselect(
                "Extrusion stages", available_orifices,
                default=available_orifices,
                key="mixing_plot_orifices",
            )
            mixing_view = mixing_pressure_table[
                mixing_pressure_table["Extruder"].eq(selected_mixing_extruder)
                & mixing_pressure_table["Orifice"].isin(selected_orifices)
            ].copy()
            st.dataframe(
                mixing_view,
                use_container_width=True,
                hide_index=True,
                height=360,
            )

            batch_number_for_mixing = batch.get("Batch no.") or "Batch"
            for orifice in selected_orifices:
                mixing_figure = create_mixing_pressure_plot(
                    mixing_pressure_table,
                    selected_mixing_extruder,
                    orifice,
                    batch_number_for_mixing,
                )
                st.plotly_chart(
                    mixing_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": (
                                f"{batch_number_for_mixing}_extruder_"
                                f"{selected_mixing_extruder}_{orifice.replace(' ', '_')}_max_pressure"
                            ),
                            "height": 620,
                            "width": 1200,
                            "scale": 2,
                        },
                    },
                    key=f"mixing_chart_{selected_mixing_extruder}_{orifice}",
                )

            mixing_html_figure = go.Figure()
            for orifice in selected_orifices:
                source_figure = create_mixing_pressure_plot(
                    mixing_pressure_table,
                    selected_mixing_extruder,
                    orifice,
                    batch_number_for_mixing,
                )
                for trace in source_figure.data:
                    trace.name = f"{orifice} | {trace.name}"
                    mixing_html_figure.add_trace(trace)
            mixing_html_figure.update_layout(
                title=f"{batch_number_for_mixing} | Extruder {selected_mixing_extruder} | Mixing Max Pressure",
                xaxis_title="Cycles", yaxis_title="Maximum Pressure (bar)",
                hovermode="closest", height=700,
            )
            mixing_html = pio.to_html(
                mixing_html_figure, include_plotlyjs="cdn", full_html=True
            ).encode("utf-8")
            m1, m2 = st.columns(2)
            m1.download_button(
                "Download Mixing Max-Pressure Table CSV",
                mixing_view.to_csv(index=False).encode("utf-8"),
                f"{batch_number_for_mixing}_extruder_{selected_mixing_extruder}_mixing_max_pressure.csv",
                "text/csv",
            )
            m2.download_button(
                "Download Interactive Mixing Plot HTML",
                mixing_html,
                f"{batch_number_for_mixing}_extruder_{selected_mixing_extruder}_mixing_max_pressure.html",
                "text/html",
            )

    with blending_tab:
        st.markdown("#### Blending Maximum Pressure by Cycle")
        st.caption(
            "Pre-pH Blending contains 60 cycles. Post-pH Blending begins when the "
            "cycle counter resets after completed cycle 60 and contains up to 100 cycles. "
            "Movement to accumulator right uses maximum Loadcell Left pressure; movement "
            "to accumulator left uses maximum Loadcell Right pressure."
        )
        if blending_pressure_table.empty:
            st.warning("No directional Blending pressure-cycle data were generated.")
        else:
            blending_extruders = sorted(
                blending_pressure_table["Extruder"].dropna().astype(int).unique()
            )
            selected_blending_extruder = st.selectbox(
                "Blending extruder", blending_extruders,
                key="blending_plot_extruder",
            )
            available_phases = [
                phase for phase in ["Pre-pH Blending", "Post-pH Blending"]
                if phase in set(
                    blending_pressure_table.loc[
                        blending_pressure_table["Extruder"].eq(selected_blending_extruder),
                        "Blending Phase",
                    ].astype(str)
                )
            ]
            selected_phases = st.multiselect(
                "Blending phases", available_phases,
                default=available_phases,
                key="blending_plot_phases",
            )
            blending_view = blending_pressure_table[
                blending_pressure_table["Extruder"].eq(selected_blending_extruder)
                & blending_pressure_table["Blending Phase"].astype(str).isin(selected_phases)
            ].copy()
            st.dataframe(
                blending_view, use_container_width=True,
                hide_index=True, height=380,
            )

            batch_number_for_blending = batch.get("Batch no.") or "Batch"
            for phase in selected_phases:
                blending_figure = create_blending_pressure_plot(
                    blending_pressure_table,
                    selected_blending_extruder,
                    phase,
                    batch_number_for_blending,
                )
                st.plotly_chart(
                    blending_figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": (
                                f"{batch_number_for_blending}_extruder_"
                                f"{selected_blending_extruder}_{phase.replace(' ', '_')}_max_pressure"
                            ),
                            "height": 620, "width": 1200, "scale": 2,
                        },
                    },
                    key=f"blending_chart_{selected_blending_extruder}_{phase}",
                )

            combined_blending_figure = go.Figure()
            for phase in selected_phases:
                source_figure = create_blending_pressure_plot(
                    blending_pressure_table,
                    selected_blending_extruder,
                    phase,
                    batch_number_for_blending,
                )
                for trace in source_figure.data:
                    trace.name = f"{phase} | {trace.name}"
                    combined_blending_figure.add_trace(trace)
            combined_blending_figure.update_layout(
                title=(
                    f"{batch_number_for_blending} | Extruder "
                    f"{selected_blending_extruder} | Blending Max Pressure"
                ),
                xaxis_title="Cycles", yaxis_title="Maximum Pressure (bar)",
                hovermode="closest", height=700,
            )
            blending_html = pio.to_html(
                combined_blending_figure,
                include_plotlyjs="cdn", full_html=True,
            ).encode("utf-8")
            b1, b2 = st.columns(2)
            b1.download_button(
                "Download Blending Max-Pressure Table CSV",
                blending_view.to_csv(index=False).encode("utf-8"),
                f"{batch_number_for_blending}_extruder_{selected_blending_extruder}_blending_max_pressure.csv",
                "text/csv",
            )
            b2.download_button(
                "Download Interactive Blending Plot HTML",
                blending_html,
                f"{batch_number_for_blending}_extruder_{selected_blending_extruder}_blending_max_pressure.html",
                "text/html",
            )

    with premixing_tab:
        st.markdown("#### Premixing Predicted Volume-Step Analysis")
        st.caption(
            "Cylinder diameter = 80.1 mm for all extruders. Direction is inferred from "
            "servo-speed signs. Opposite movement blocks are paired using same-cycle "
            "preference. Volume step is reconstructed dynamically from each batch's "
            "logged-cycle sequence and actual maximum-pressure increases. Any positive "
            "pressure increase is recognized; no pressure tolerance or fixed attempt count is used."
        )
        if premix_summary.empty:
            st.warning("No Premixing bidirectional volume-step analysis was generated.")
        else:
            premix_extruders = sorted(premix_summary["Extruder"].unique().astype(int))
            selected_premix_extruder = st.selectbox(
                "Premixing extruder", premix_extruders,
                key="premixing_analysis_extruder",
            )
            summary_view = premix_summary[
                premix_summary["Extruder"].eq(selected_premix_extruder)
            ].copy()
            batch_number_for_premix = batch.get("Batch no.") or "Batch"
            premix_figure = create_premixing_volume_step_plot(
                premix_summary,
                selected_premix_extruder,
                batch_number_for_premix,
            )
            st.plotly_chart(
                premix_figure,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": (
                            f"{batch_number_for_premix}_extruder_"
                            f"{selected_premix_extruder}_premixing_volume_step"
                        ),
                        "height": 700, "width": 1200, "scale": 2,
                    },
                },
                key=f"premix_chart_{selected_premix_extruder}",
            )
            st.dataframe(
                summary_view, use_container_width=True,
                hide_index=True, height=360,
            )

            with st.expander("Directional movement blocks"):
                st.dataframe(
                    premix_blocks[
                        premix_blocks["Extruder"].eq(selected_premix_extruder)
                    ],
                    use_container_width=True, hide_index=True, height=420,
                )
            with st.expander("Improved bidirectional pairing"):
                st.dataframe(
                    premix_pairs[
                        premix_pairs["Extruder"].eq(selected_premix_extruder)
                    ],
                    use_container_width=True, hide_index=True, height=420,
                )
            with st.expander("Reverse-predicted volume and pressure sequence"):
                st.dataframe(
                    premix_prediction[
                        premix_prediction["Extruder"].eq(selected_premix_extruder)
                    ],
                    use_container_width=True, hide_index=True, height=450,
                )

            premix_html = pio.to_html(
                premix_figure, include_plotlyjs="cdn", full_html=True
            ).encode("utf-8")
            p1, p2 = st.columns(2)
            p1.download_button(
                "Download Premixing Volume-Step Summary CSV",
                summary_view.to_csv(index=False).encode("utf-8"),
                f"{batch_number_for_premix}_extruder_{selected_premix_extruder}_premixing_summary.csv",
                "text/csv",
            )
            p2.download_button(
                "Download Interactive Premixing Plot HTML",
                premix_html,
                f"{batch_number_for_premix}_extruder_{selected_premix_extruder}_premixing_plot.html",
                "text/html",
            )

    with diagnostics_tab:
        st.markdown("#### Assigned process page ranges"); st.dataframe(process_ranges, use_container_width=True, hide_index=True)
        st.markdown("#### Page-level diagnostics"); st.dataframe(diagnostics, use_container_width=True, hide_index=True, height=400)
        st.markdown("#### Rejected rows")
        if rejected_rows.empty: st.success("No timestamp rows were rejected.")
        else: st.dataframe(rejected_rows, use_container_width=True, hide_index=True, height=400)

    excel = make_excel(batch, overview, summaries, settings, datalog, diagnostics, rejected_rows, process_ranges, wetting_rise_table, mixing_pressure_table, blending_pressure_table, premix_prepared, premix_blocks, premix_pairs, premix_prediction, premix_summary)
    batch_number = batch.get("Batch no.") or "batch"
    st.download_button("⬇️ Download Complete Excel Extraction", excel, f"{batch_number}_complete_process_extraction.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")


if __name__ == "__main__":
    main()
