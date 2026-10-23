import io
import math
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Batch Process Data Extractor", page_icon="🧪", layout="wide")

STAGES = ("WETTING", "PREMIXING", "MIXING", "BLENDING")
MAX_UPLOAD_FILES = 20
CACHE_TTL_SECONDS = 1800
CACHE_MAX_PAGES = 5
CACHE_MAX_REPORTS = 3
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
PREMIXING_PRESSURE_MATCH_TOLERANCE_BAR = 0.5
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


@st.cache_data(
    show_spinner=False,
    ttl=CACHE_TTL_SECONDS,
    max_entries=CACHE_MAX_PAGES,
)
def extract_pages(pdf_bytes):
    parser_version = "2026.10.23-20-files-no-size-limit-v24"
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


def normalize_premixing_pressure(actual_pressure):
    """Normalize pressure: decimal <= 0.5 rounds down; > 0.5 rounds up."""
    if pd.isna(actual_pressure):
        return np.nan
    pressure = float(actual_pressure)
    lower_bar = math.floor(pressure)
    return int(lower_bar if pressure - lower_bar <= 0.5 else lower_bar + 1)


def expected_premixing_left_volume(volume_step_ml, step_volume_ml):
    """Standard left-pushed volume for a cumulative Premixing volume step."""
    if pd.isna(volume_step_ml):
        return np.nan
    return 2 * float(volume_step_ml) - float(step_volume_ml)


def premixing_recipe_pressure_step(actual_pressure):
    """Map the actual maximum pressure to the nearest 15-bar recipe step."""
    if pd.isna(actual_pressure):
        return np.nan
    normalized = normalize_premixing_pressure(actual_pressure)
    if normalized <= 0:
        return 0.0
    step_number = max(1, int(np.floor(
        (normalized + PREMIXING_PRESSURE_STEP_BAR / 2)
        / PREMIXING_PRESSURE_STEP_BAR
    )))
    return float(min(
        step_number * PREMIXING_PRESSURE_STEP_BAR,
        PREMIXING_MAX_PRESSURE_BAR,
    ))


def reverse_predict_premixing_volume_step(paired_attempts, step_volume_ml=PREMIXING_VOLUME_STEP_ML):
    """Reconstruct volume step using cycle, normalized pressure, and left volume."""
    if paired_attempts.empty:
        return pd.DataFrame()

    step_volume_ml = float(step_volume_ml)
    if step_volume_ml <= 0:
        raise ValueError("Premixing step volume must be greater than 0 mL.")

    output = []
    for extruder, attempts in paired_attempts.groupby("Extruder", sort=True):
        volume_step_number = 1
        volume_step_size = step_volume_ml
        highest_normalized_pressure = int(PREMIXING_PRESSURE_STEP_BAR)
        established_recipe_pressure = PREMIXING_PRESSURE_STEP_BAR
        previous_cycle = None

        ordered = attempts.sort_values("Start Date & Time", kind="stable")
        for attempt_number, (_, row) in enumerate(ordered.iterrows(), start=1):
            actual_cycle = row["Actual Logged Cycle"]
            current_cycle = int(actual_cycle) if pd.notna(actual_cycle) else None
            actual_pressure = pd.to_numeric(
                row["Maximum Actual Pressure (bar)"], errors="coerce"
            )
            normalized_pressure = normalize_premixing_pressure(actual_pressure)
            previous_highest = highest_normalized_pressure
            pressure_increased = (
                pd.notna(normalized_pressure)
                and normalized_pressure > previous_highest
            )
            inferred_recipe_pressure = premixing_recipe_pressure_step(actual_pressure)
            if pd.notna(inferred_recipe_pressure):
                established_recipe_pressure = max(
                    established_recipe_pressure, inferred_recipe_pressure
                )

            actual_left_volume = pd.to_numeric(
                row["L-to-R Pushed Volume (mL)"], errors="coerce"
            )
            expected_left_volume = expected_premixing_left_volume(volume_step_size, step_volume_ml)
            left_difference = (
                actual_left_volume - expected_left_volume
                if pd.notna(actual_left_volume) and pd.notna(expected_left_volume)
                else np.nan
            )
            pushed_volume_within_target = (
                pd.notna(actual_left_volume)
                and pd.notna(expected_left_volume)
                and actual_left_volume <= expected_left_volume + 1.0
            )

            if attempt_number == 1:
                rule = "Initial volume step"
            elif current_cycle is not None and previous_cycle is not None and current_cycle > previous_cycle:
                rule = "Same volume step: higher cycle repeat attempt"
            elif previous_cycle == PREMIXING_REPEAT_CYCLES and current_cycle == 1:
                if pressure_increased:
                    rule = "Same volume step: Cycle 3 to 1 pressure increase"
                elif pushed_volume_within_target:
                    rule = "Same volume step: L-to-R pushed volume <= target volume + 1"
                else:
                    volume_step_number += 1
                    volume_step_size = volume_step_number * step_volume_ml
                    rule = "Advanced by 16.5 mL: no pressure increase and L-to-R pushed volume > target volume + 1"
            elif current_cycle == 1:
                volume_step_number += 1
                volume_step_size = volume_step_number * step_volume_ml
                rule = "Advanced by 16.5 mL: Cycle 1 did not increase"
            else:
                rule = "Same volume step: cycle sequence review"

            if pd.notna(normalized_pressure):
                highest_normalized_pressure = max(
                    highest_normalized_pressure, normalized_pressure
                )

            output.append({
                **row.to_dict(),
                "Attempt Number": attempt_number,
                "Previous Logged Cycle": previous_cycle,
                "Volume step size (mL)": volume_step_size,
                "Volume Step Number": volume_step_number,
                "Selected Premixing Step Volume (mL)": step_volume_ml,
                "Volume Step Rule": rule,
                "Actual Maximum Pressure (bar)": actual_pressure,
                "Normalized Maximum Pressure (bar)": normalized_pressure,
                "Pressure Step (bar)": established_recipe_pressure,
                "Previous Highest Normalized Pressure (bar)": previous_highest,
                "Highest Normalized Pressure Seen (bar)": highest_normalized_pressure,
                "Pressure Increased": "Yes" if pressure_increased else "No",
                "Expected Left Volume (mL)": expected_left_volume,
                "Actual Left Volume (mL)": actual_left_volume,
                "Left Volume Difference (mL)": left_difference,
                "L-to-R Pushed Volume <= Target Volume + 1": (
                    "Yes" if pushed_volume_within_target else "No"
                ),
            })
            if current_cycle is not None:
                previous_cycle = current_cycle

    return pd.DataFrame(output)


def build_premixing_volume_step_summary(prediction):
    """Summarize attempts, actual pressures, and recipe pressure step by volume step."""
    columns = [
        "Extruder", "Volume Step (mL)", "Attempts",
        "Max Loadcell Left (bar)", "Max Loadcell Right (bar)",
        "Pressure Step (bar)", "First Attempt", "Last Attempt",
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
            "Max Loadcell Left (bar)": group["L-to-R Maximum Push Pressure (bar)"].max(),
            "Max Loadcell Right (bar)": group["R-to-L Maximum Push Pressure (bar)"].max(),
            "Pressure Step (bar)": group["Pressure Step (bar)"].max(),
            "First Attempt": group["Start Date & Time"].min(),
            "Last Attempt": group["End Date & Time"].max(),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Extruder", "Volume Step (mL)"]
    ).reset_index(drop=True)


def create_premixing_volume_step_plot(
    summary, extruder, batch_number, pressure_mode="Actual Max Pressure",
    step_volume_ml=PREMIXING_VOLUME_STEP_ML,
):
    """Create Premixing pressure/attempt plot with fixed 0-198 mL x-axis."""
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
        x=data["Volume Step (mL)"], y=data["Attempts"],
        name="Attempts", yaxis="y2",
        width=float(step_volume_ml) * 0.62,
        marker={"color": "rgba(181, 211, 143, 0.55)",
                "line": {"color": "rgba(123, 158, 83, 0.85)", "width": 1}},
        customdata=date_details.to_numpy(),
        hovertemplate=(
            "<b>Attempts</b><br>Volume step: %{x:.1f} mL<br>"
            "Number of attempts: %{y:.0f}<br>First attempt: %{customdata[0]}<br>"
            "Last attempt: %{customdata[1]}<extra></extra>"
        ),
    ))

    if pressure_mode == "Pressure Step":
        figure.add_trace(go.Scatter(
            x=data["Volume Step (mL)"], y=data["Pressure Step (bar)"],
            mode="lines+markers", name="Pressure Step (bar)",
            line={"color": "#7030A0", "width": 4, "shape": "hv"},
            marker={"size": 9, "color": "#7030A0", "symbol": "square"},
            hovertemplate=(
                "<b>Pressure Step</b><br>Volume step: %{x:.1f} mL<br>"
                "Recipe pressure step: %{y:.0f} bar<extra></extra>"
            ),
        ))
    else:
        figure.add_trace(go.Scatter(
            x=data["Volume Step (mL)"], y=data["Max Loadcell Left (bar)"],
            mode="lines+markers", name="Max Loadcell Left (bar)",
            line={"color": "#0057B8", "width": 4, "shape": "hv"},
            marker={"size": 9, "color": "#0057B8", "symbol": "circle"},
            hovertemplate=(
                "<b>Max Loadcell Left</b><br>Volume step: %{x:.1f} mL<br>"
                "Maximum pressure: %{y:.2f} bar<extra></extra>"
            ),
        ))
        figure.add_trace(go.Scatter(
            x=data["Volume Step (mL)"], y=data["Max Loadcell Right (bar)"],
            mode="lines+markers", name="Max Loadcell Right (bar)",
            line={"color": "#D62728", "width": 4, "shape": "hv"},
            marker={"size": 9, "color": "#D62728", "symbol": "diamond"},
            hovertemplate=(
                "<b>Max Loadcell Right</b><br>Volume step: %{x:.1f} mL<br>"
                "Maximum pressure: %{y:.2f} bar<extra></extra>"
            ),
        ))

    max_attempts = int(data["Attempts"].max()) if not data.empty else 1
    figure.update_layout(
        title={"text": f"{batch_number} | EXTRUDER-{int(extruder)} PRE-MIXING",
               "x": 0.5, "xanchor": "center", "y": 0.97, "yanchor": "top",
               "font": {"family": "Times New Roman", "size": 12, "color": "#000000"}},
        font={"family": "Times New Roman", "size": 12, "color": "#000000"},
        xaxis={"title": {"text": "<b>VOLUME STEP (mL)</b>", "standoff": 18},
               "range": [0, float(step_volume_ml) * 12], "dtick": float(step_volume_ml),
               "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
        yaxis={"title": {"text": "<b>PRESSURE (bar)</b>", "standoff": 12},
               "range": [0, 200], "dtick": 15 if pressure_mode == "Pressure Step" else 20,
               "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
        yaxis2={"title": {"text": "<b>ATTEMPTS</b>", "standoff": 12},
                "overlaying": "y", "side": "right",
                "range": [0, max(4, max_attempts + 1)], "dtick": 1, "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1},
        barmode="overlay", hovermode="x unified",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center",
                "y": 1.03, "yanchor": "bottom",
                "font": {"family": "Times New Roman", "size": 12},
                "bgcolor": "rgba(255,255,255,0.9)"},
        plot_bgcolor="white", paper_bgcolor="white", height=720,
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
        xaxis={"range": [0, expected_cycles + 2], "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
        yaxis={"range": [0, y_max], "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
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
        xaxis={"dtick": 1, "range": [0.5, max_cycle + 0.5], "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
        yaxis={"range": [0, y_max], "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
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
        xaxis={"range": [0, x_max], "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
        yaxis={"range": [0, max(700.0, limit * 1.15)], "showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
        hovermode="closest",
        legend={"x": 0.72, "y": 0.45, "bgcolor": "rgba(255,255,255,0.65)"},
        height=700,
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
    )
    return figure


def figure_to_png(fig):
    buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight"); return buffer.getvalue()


def apply_graph_style(figure, font_family="Times New Roman", font_size=12):
    """Apply typography, remove grids, and draw the inner plot border."""
    size = int(font_size)
    figure.update_layout(
        font={"family": font_family, "size": size, "color": "#000000"},
        title_font={"family": font_family, "size": size, "color": "#000000"},
        legend_font={"family": font_family, "size": size, "color": "#000000"},
        paper_bgcolor="white", plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#C7C7C7", gridwidth=1,
        showline=True, mirror=True, linewidth=1.5,
        linecolor="#000000", ticks="outside",
        tickfont={"family": font_family, "size": size, "color": "#000000"},
        title_font={"family": font_family, "size": size, "color": "#000000"})
    figure.update_yaxes(showgrid=True, gridcolor="#C7C7C7", gridwidth=1,
        showline=True, mirror=True, linewidth=1.5,
        linecolor="#000000", ticks="outside",
        tickfont={"family": font_family, "size": size, "color": "#000000"},
        title_font={"family": font_family, "size": size, "color": "#000000"})
    return figure


def graph_display_controls():
    """Sidebar font controls shared by every graph."""
    with st.sidebar:
        st.markdown("### Application Controls")
        if st.button("Clear processed reports and cache", use_container_width=True):
            st.cache_data.clear()
            generated_prefixes = (
                "generated_excel_", "generated_zip_", "generated_html_"
            )
            for state_key in list(st.session_state):
                if state_key.startswith(generated_prefixes):
                    del st.session_state[state_key]
            st.rerun()
        st.caption(
            "Use this before uploading a completely new report set if the app "
            "becomes slow."
        )
        st.markdown("### Graph Controls")
        family = st.selectbox("Graph font", ["Times New Roman", "Arial", "Calibri", "Georgia"], index=0, key="graph_font_family")
        preset = st.selectbox("Graph font size", ["10 pt", "12 pt", "14 pt", "16 pt", "18 pt", "20 pt", "Custom"], index=1, key="graph_font_size_preset")
        if preset == "Custom":
            size = int(st.number_input("Custom font size", min_value=8, max_value=30, value=12, step=1, key="graph_custom_font_size"))
        else:
            size = int(preset.split()[0])
    return family, size


def create_editable_interactive_html(figure, x_axis_label="X axis"):
    """Export HTML with movable labels and manual X/Y limits and font controls."""
    html_figure = go.Figure(figure)
    html_figure.update_xaxes(rangeslider={"visible": False}, fixedrange=False)
    html_figure.update_yaxes(fixedrange=False)
    apply_graph_style(html_figure, html_figure.layout.font.family or "Times New Roman", html_figure.layout.font.size or 12)
    config = {
        "editable": True,
        "edits": {"annotationPosition": True, "annotationText": True,
                  "axisTitleText": True, "legendPosition": True,
                  "legendText": True, "shapePosition": True, "titleText": True},
        "displaylogo": False, "scrollZoom": True, "responsive": True,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }
    html = pio.to_html(html_figure, include_plotlyjs=True, full_html=True,
                       config=config, div_id="editable_plot")
    controls = """
<style>
body { margin:0; background:#fff; font-family:'Times New Roman',serif; }
#graph_editor_panel { margin:10px auto; padding:10px 12px; width:calc(100% - 44px); border:1.5px solid #000; background:#f7f7f7; box-sizing:border-box; display:flex; flex-wrap:wrap; gap:8px 12px; align-items:end; }
#graph_editor_panel label { font-weight:bold; font-size:12px; }
#graph_editor_panel input,#graph_editor_panel select,#graph_editor_panel button { font-family:inherit; font-size:12px; padding:4px 6px; }
#editable_plot { margin:10px 20px 20px 20px; border:2px solid #000; box-sizing:border-box; }
.editor_field { display:flex; flex-direction:column; gap:2px; }
.grid_option { display:flex; align-items:center; gap:5px; padding-bottom:5px; }
</style>
<div id="graph_editor_panel">
<div class="editor_field"><label>X minimum</label><input id="x_min" type="number" step="any"></div>
<div class="editor_field"><label>X maximum</label><input id="x_max" type="number" step="any"></div>
<div class="editor_field"><label>Y minimum</label><input id="y_min" type="number" step="any"></div>
<div class="editor_field"><label>Y maximum</label><input id="y_max" type="number" step="any"></div>
<div class="editor_field"><label>Font</label><select id="font_family"><option>Times New Roman</option><option>Arial</option><option>Calibri</option><option>Georgia</option></select></div>
<div class="editor_field"><label>Font size</label><input id="font_size" type="number" min="8" max="40" value="12"></div>
<div class="grid_option"><input id="x_grid" type="checkbox" checked><label for="x_grid">X-axis gridlines</label></div>
<div class="grid_option"><input id="y_grid" type="checkbox" checked><label for="y_grid">Y-axis gridlines</label></div>
<div class="grid_option"><input id="y2_grid" type="checkbox"><label for="y2_grid">Secondary Y-axis gridlines</label></div>
<button id="apply_graph_settings">Apply</button><button id="auto_scale">Auto scale</button>
<span><b>Tip:</b> drag the title, legend and annotations directly in the graph.</span>
</div>
<script>
window.addEventListener('load',function(){
 const plot=document.getElementById('editable_plot'),layout=plot.layout||{};
 const xr=layout.xaxis&&layout.xaxis.range?layout.xaxis.range:[],yr=layout.yaxis&&layout.yaxis.range?layout.yaxis.range:[];
 if(xr.length===2){x_min.value=xr[0];x_max.value=xr[1];} if(yr.length===2){y_min.value=yr[0];y_max.value=yr[1];}
 if(layout.font&&layout.font.family)font_family.value=layout.font.family; if(layout.font&&layout.font.size)font_size.value=layout.font.size;
 x_grid.checked = !(layout.xaxis && layout.xaxis.showgrid === false);
 y_grid.checked = !(layout.yaxis && layout.yaxis.showgrid === false);
 y2_grid.checked = !!(layout.yaxis2 && layout.yaxis2.showgrid === true);
 if (!(layout.yaxis2)) { y2_grid.disabled = true; y2_grid.parentElement.style.opacity = '0.5'; }
 apply_graph_settings.onclick=function(){
  const xmin=parseFloat(x_min.value),xmax=parseFloat(x_max.value),ymin=parseFloat(y_min.value),ymax=parseFloat(y_max.value),family=font_family.value,size=parseInt(font_size.value||'12');
  const u={'font.family':family,'font.size':size,'title.font.family':family,'title.font.size':size,'legend.font.family':family,'legend.font.size':size,'xaxis.tickfont.family':family,'xaxis.tickfont.size':size,'xaxis.title.font.family':family,'xaxis.title.font.size':size,'yaxis.tickfont.family':family,'yaxis.tickfont.size':size,'yaxis.title.font.family':family,'yaxis.title.font.size':size,'xaxis.showgrid':x_grid.checked,'yaxis.showgrid':y_grid.checked};
  if(layout.yaxis2) u['yaxis2.showgrid']=y2_grid.checked;
  if(!Number.isNaN(xmin)&&!Number.isNaN(xmax)&&xmin<xmax)u['xaxis.range']=[xmin,xmax]; if(!Number.isNaN(ymin)&&!Number.isNaN(ymax)&&ymin<ymax)u['yaxis.range']=[ymin,ymax]; Plotly.relayout(plot,u);
 };
 auto_scale.onclick=function(){Plotly.relayout(plot,{'xaxis.autorange':true,'yaxis.autorange':true});x_min.value='';x_max.value='';y_min.value='';y_max.value='';};
});
</script>
"""
    return html.replace('<body>', '<body>'+controls).encode('utf-8')


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



@st.cache_data(
    show_spinner=False,
    ttl=CACHE_TTL_SECONDS,
    max_entries=CACHE_MAX_REPORTS,
)
def process_uploaded_report(file_name, pdf_bytes, premixing_step_volume_ml):
    """Extract and calculate all report outputs for one uploaded PDF."""
    pages = extract_pages(pdf_bytes)
    batch = parse_batch_details(pages)
    overview = parse_process_overview(pages)
    summaries = parse_process_summaries(pages)
    settings = parse_recipe_settings(pages, summaries)
    datalog, diagnostics, rejected_rows, process_ranges = parse_datalogs(
        pages, summaries
    )

    wetting_data = (
        datalog[datalog["Stage"].eq("WETTING")].copy()
        if not datalog.empty else pd.DataFrame()
    )
    wetting_rise_table = build_wetting_rise_time_table(wetting_data)

    mixing_data = (
        datalog[datalog["Stage"].eq("MIXING")].copy()
        if not datalog.empty else pd.DataFrame()
    )
    mixing_pressure_table = build_mixing_max_pressure_table(mixing_data)

    blending_data = (
        datalog[datalog["Stage"].eq("BLENDING")].copy()
        if not datalog.empty else pd.DataFrame()
    )
    blending_pressure_table = build_blending_max_pressure_table(blending_data)

    premixing_data = (
        datalog[datalog["Stage"].eq("PREMIXING")].copy()
        if not datalog.empty else pd.DataFrame()
    )
    premix_prepared = prepare_premixing_engineering_data(premixing_data)
    premix_blocks = summarize_premixing_directional_blocks(premix_prepared)
    premix_pairs = pair_premixing_bidirectional_attempts(premix_blocks)
    premix_prediction = reverse_predict_premixing_volume_step(
        premix_pairs, premixing_step_volume_ml
    )

    if not premix_pairs.empty and not premix_prediction.empty:
        volume_step_map = premix_prediction[[
            "Extruder", "Bidirectional Attempt ID",
            "Volume step size (mL)", "Volume Step Rule",
        ]].drop_duplicates(["Extruder", "Bidirectional Attempt ID"])
        premix_pairs = premix_pairs.merge(
            volume_step_map,
            on=["Extruder", "Bidirectional Attempt ID"],
            how="left",
        )

    premix_summary = build_premixing_volume_step_summary(premix_prediction)

    return {
        "file_name": file_name,
        "pages": pages,
        "batch": batch,
        "overview": overview,
        "summaries": summaries,
        "settings": settings,
        "datalog": datalog,
        "diagnostics": diagnostics,
        "rejected_rows": rejected_rows,
        "process_ranges": process_ranges,
        "wetting_rise_table": wetting_rise_table,
        "mixing_pressure_table": mixing_pressure_table,
        "blending_pressure_table": blending_pressure_table,
        "premix_prepared": premix_prepared,
        "premix_blocks": premix_blocks,
        "premix_pairs": premix_pairs,
        "premix_prediction": premix_prediction,
        "premix_summary": premix_summary,
        "premixing_step_volume_ml": float(premixing_step_volume_ml),
    }


def result_to_excel(result):
    """Create the complete Excel workbook for one processed report."""
    return make_excel(
        result["batch"], result["overview"], result["summaries"],
        result["settings"], result["datalog"], result["diagnostics"],
        result["rejected_rows"], result["process_ranges"],
        result["wetting_rise_table"], result["mixing_pressure_table"],
        result["blending_pressure_table"], result["premix_prepared"],
        result["premix_blocks"], result["premix_pairs"],
        result["premix_prediction"], result["premix_summary"],
    )


def safe_output_name(value):
    """Create a filesystem-safe output name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "batch"


def build_all_batches_zip(results, errors):
    """Package one complete Excel workbook per successful PDF into one ZIP."""
    output = io.BytesIO()
    used_names = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result_key, result in results.items():
            batch_number = result["batch"].get("Batch no.") or Path(
                result["file_name"]
            ).stem
            base_name = safe_output_name(batch_number)
            candidate = f"{base_name}_complete_process_extraction.xlsx"
            suffix = 2
            while candidate.casefold() in used_names:
                candidate = (
                    f"{base_name}_{suffix}_complete_process_extraction.xlsx"
                )
                suffix += 1
            used_names.add(candidate.casefold())
            archive.writestr(candidate, result_to_excel(result))

        if errors:
            error_lines = [
                f"{file_name}: {message}"
                for file_name, message in errors.items()
            ]
            archive.writestr(
                "extraction_errors.txt",
                "\n".join(error_lines).encode("utf-8"),
            )
    return output.getvalue()

def main():
    st.title("🧪 Complete Batch Process Report Extractor")
    st.caption(
        "Upload and analyze one or more batch-cycle PDF reports. Each file is "
        "processed independently so a failure in one report does not stop the others."
    )
    graph_font_family, graph_font_size = graph_display_controls()

    st.markdown("### Premixing batch configuration")
    premixing_batch_size = st.selectbox(
        "Premixing batch size / step volume",
        ["1 kg batch - 16.5 mL", "1.5 kg batch - 25.0 mL", "Custom"],
        index=0,
    )
    if premixing_batch_size == "1 kg batch - 16.5 mL":
        premixing_step_volume_ml = 16.5
    elif premixing_batch_size == "1.5 kg batch - 25.0 mL":
        premixing_step_volume_ml = 25.0
    else:
        premixing_step_volume_ml = float(st.number_input(
            "Custom Premixing step volume (mL)", min_value=0.1,
            max_value=100.0, value=16.5, step=0.5, format="%.1f"
        ))
    st.info(
        f"Selected Premixing step volume: {premixing_step_volume_ml:g} mL | "
        f"Plot range: 0 to {premixing_step_volume_ml * 12:g} mL"
    )

    uploaded_files = st.file_uploader(
        "Upload Batch Cycle Report PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF batch reports.",
    )
    if not uploaded_files:
        st.info("Upload one or more batch-report PDFs to begin.")
        st.stop()

    if len(uploaded_files) > MAX_UPLOAD_FILES:
        st.error(
            f"Upload a maximum of {MAX_UPLOAD_FILES} PDF reports per processing run."
        )
        st.stop()

    st.caption(
        f"Selected {len(uploaded_files)} file(s). "
        f"Maximum allowed per processing run: {MAX_UPLOAD_FILES} files."
    )

    processed_results = {}
    extraction_errors = {}
    progress = st.progress(0, text="Preparing report extraction...")

    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        unique_key = f"{file_index:03d}_{uploaded_file.name}"
        progress.progress(
            (file_index - 1) / len(uploaded_files),
            text=f"Processing {file_index}/{len(uploaded_files)}: {uploaded_file.name}",
        )
        try:
            processed_results[unique_key] = process_uploaded_report(
                uploaded_file.name,
                uploaded_file.getvalue(),
                premixing_step_volume_ml,
            )
        except Exception as error:
            extraction_errors[uploaded_file.name] = str(error)

    progress.progress(1.0, text="All uploaded reports have been processed.")

    if extraction_errors:
        with st.expander(
            f"Files requiring review ({len(extraction_errors)})",
            expanded=True,
        ):
            for file_name, message in extraction_errors.items():
                st.error(f"{file_name}: {message}")

    if not processed_results:
        st.error("None of the uploaded PDF reports could be processed.")
        st.stop()

    summary_rows = []
    selector_labels = {}
    for result_key, result in processed_results.items():
        batch_number = result["batch"].get("Batch no.") or "Batch number not found"
        selector_label = f"{batch_number} | {result['file_name']}"
        selector_labels[selector_label] = result_key
        summary_rows.append({
            "File Name": result["file_name"],
            "Batch No.": batch_number,
            "PDF Pages": len(result["pages"]),
            "Process Runs": len(result["summaries"]),
            "Datalog Rows": len(result["datalog"]),
            "Rejected Rows": len(result["rejected_rows"]),
            "Premixing Step (mL)": result["premixing_step_volume_ml"],
            "Status": "Processed",
        })

    st.markdown("### Uploaded report summary")
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
    )

    selected_label = st.selectbox(
        "Select report to review",
        options=list(selector_labels),
        key="selected_batch_report",
    )
    selected_result = processed_results[selector_labels[selected_label]]

    pages = selected_result["pages"]
    batch = selected_result["batch"]
    overview = selected_result["overview"]
    summaries = selected_result["summaries"]
    settings = selected_result["settings"]
    datalog = selected_result["datalog"]
    diagnostics = selected_result["diagnostics"]
    rejected_rows = selected_result["rejected_rows"]
    process_ranges = selected_result["process_ranges"]
    wetting_rise_table = selected_result["wetting_rise_table"]
    mixing_pressure_table = selected_result["mixing_pressure_table"]
    blending_pressure_table = selected_result["blending_pressure_table"]
    premix_prepared = selected_result["premix_prepared"]
    premix_blocks = selected_result["premix_blocks"]
    premix_pairs = selected_result["premix_pairs"]
    premix_prediction = selected_result["premix_prediction"]
    premix_summary = selected_result["premix_summary"]
    selected_premixing_step_volume_ml = selected_result["premixing_step_volume_ml"]

    st.markdown(f"### Selected report: {selected_result['file_name']}")
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
            apply_graph_style(interactive_figure, graph_font_family, graph_font_size)
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
            x1, x2 = st.columns(2)
            x1.download_button(
                "Download Rise-Time Table CSV",
                table_view.to_csv(index=False).encode(),
                f"{batch_number}_wetting_rise_time.csv",
                "text/csv",
            )
            wetting_html_key = (
                f"generated_html_wetting_{selected_label}_{','.join(map(str, selected))}"
            )
            if x2.button("Prepare Interactive Plot HTML", key=f"prepare_{wetting_html_key}"):
                with st.spinner("Preparing interactive Wetting HTML..."):
                    st.session_state[wetting_html_key] = create_editable_interactive_html(
                        interactive_figure, "Elapsed time"
                    )
            if wetting_html_key in st.session_state:
                x2.download_button(
                    "Download Interactive Plot HTML",
                    st.session_state[wetting_html_key],
                    f"{batch_number}_wetting_rise_time_interactive.html",
                    "text/html",
                    key=f"download_{wetting_html_key}",
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
                apply_graph_style(mixing_figure, graph_font_family, graph_font_size)
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
            mixing_colors = ["#0057B8", "#D62728", "#2E8B57", "#FF8C00", "#7030A0", "#008B8B"]
            mixing_symbols = ["circle", "diamond", "square", "triangle-up", "x", "star"]
            mixing_dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
            mixing_style_index = 0
            for orifice in selected_orifices:
                source_figure = create_mixing_pressure_plot(
                    mixing_pressure_table,
                    selected_mixing_extruder,
                    orifice,
                    batch_number_for_mixing,
                )
                for trace in source_figure.data:
                    color = mixing_colors[mixing_style_index % len(mixing_colors)]
                    symbol = mixing_symbols[mixing_style_index % len(mixing_symbols)]
                    dash = mixing_dashes[mixing_style_index % len(mixing_dashes)]
                    trace.name = f"{orifice} | {trace.name}"
                    trace.line.color = color
                    trace.line.dash = dash
                    trace.line.width = 3
                    trace.marker.color = color
                    trace.marker.symbol = symbol
                    trace.marker.size = 9
                    mixing_html_figure.add_trace(trace)
                    mixing_style_index += 1
            mixing_html_figure.update_layout(
                title=f"{batch_number_for_mixing} | Extruder {selected_mixing_extruder} | Mixing Max Pressure",
                xaxis_title="<b>Cycles</b>", yaxis_title="<b>Maximum Pressure (bar)</b>",
                xaxis={"showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
                yaxis={"showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
                hovermode="closest", height=700,
            )
            apply_graph_style(mixing_html_figure, graph_font_family, graph_font_size)
            m1, m2 = st.columns(2)
            m1.download_button(
                "Download Mixing Max-Pressure Table CSV",
                mixing_view.to_csv(index=False).encode("utf-8"),
                f"{batch_number_for_mixing}_extruder_{selected_mixing_extruder}_mixing_max_pressure.csv",
                "text/csv",
            )
            mixing_html_key = (
                f"generated_html_mixing_{selected_label}_{selected_mixing_extruder}_"
                f"{'_'.join(selected_orifices)}"
            )
            if m2.button("Prepare Interactive Mixing HTML", key=f"prepare_{mixing_html_key}"):
                with st.spinner("Preparing interactive Mixing HTML..."):
                    st.session_state[mixing_html_key] = create_editable_interactive_html(
                        mixing_html_figure, "Cycles"
                    )
            if mixing_html_key in st.session_state:
                m2.download_button(
                    "Download Interactive Mixing Plot HTML",
                    st.session_state[mixing_html_key],
                    f"{batch_number_for_mixing}_extruder_{selected_mixing_extruder}_mixing_max_pressure.html",
                    "text/html",
                    key=f"download_{mixing_html_key}",
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
                apply_graph_style(blending_figure, graph_font_family, graph_font_size)
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
            blending_colors = ["#0057B8", "#D62728", "#2E8B57", "#FF8C00"]
            blending_symbols = ["circle", "diamond", "triangle-up", "square"]
            blending_dashes = ["solid", "dash", "dot", "dashdot"]
            blending_style_index = 0
            for phase in selected_phases:
                source_figure = create_blending_pressure_plot(
                    blending_pressure_table,
                    selected_blending_extruder,
                    phase,
                    batch_number_for_blending,
                )
                for trace in source_figure.data:
                    color = blending_colors[blending_style_index % len(blending_colors)]
                    symbol = blending_symbols[blending_style_index % len(blending_symbols)]
                    dash = blending_dashes[blending_style_index % len(blending_dashes)]
                    trace.name = f"{phase} | {trace.name}"
                    trace.line.color = color
                    trace.line.dash = dash
                    trace.line.width = 3
                    trace.marker.color = color
                    trace.marker.symbol = symbol
                    trace.marker.size = 9
                    combined_blending_figure.add_trace(trace)
                    blending_style_index += 1
            combined_blending_figure.update_layout(
                title=(
                    f"{batch_number_for_blending} | Extruder "
                    f"{selected_blending_extruder} | Blending Max Pressure"
                ),
                xaxis_title="<b>Cycles</b>", yaxis_title="<b>Maximum Pressure (bar)</b>",
                xaxis={"showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
                yaxis={"showgrid": True, "gridcolor": "#C7C7C7", "gridwidth": 1, },
                hovermode="closest", height=700,
            )
            apply_graph_style(combined_blending_figure, graph_font_family, graph_font_size)
            b1, b2 = st.columns(2)
            b1.download_button(
                "Download Blending Max-Pressure Table CSV",
                blending_view.to_csv(index=False).encode("utf-8"),
                f"{batch_number_for_blending}_extruder_{selected_blending_extruder}_blending_max_pressure.csv",
                "text/csv",
            )
            blending_html_key = (
                f"generated_html_blending_{selected_label}_{selected_blending_extruder}_"
                f"{'_'.join(selected_phases)}"
            )
            if b2.button("Prepare Interactive Blending HTML", key=f"prepare_{blending_html_key}"):
                with st.spinner("Preparing interactive Blending HTML..."):
                    st.session_state[blending_html_key] = create_editable_interactive_html(
                        combined_blending_figure, "Cycles"
                    )
            if blending_html_key in st.session_state:
                b2.download_button(
                    "Download Interactive Blending Plot HTML",
                    st.session_state[blending_html_key],
                    f"{batch_number_for_blending}_extruder_{selected_blending_extruder}_blending_max_pressure.html",
                    "text/html",
                    key=f"download_{blending_html_key}",
                )

    with premixing_tab:
        st.markdown("#### Premixing Predicted Volume-Step Analysis")
        st.caption(
            "Cylinder diameter = 80.1 mm for all extruders. Direction is inferred from "
            "servo-speed signs. Opposite movement blocks are paired using same-cycle "
            "preference. Select Actual Max Pressure or the inferred 15-bar Pressure Step. "
            "The Premixing x-axis is fixed from 0 to 198 mL."
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
            selected_premix_pressure_mode = st.radio(
                "Pressure to plot",
                options=["Actual Max Pressure", "Pressure Step"],
                horizontal=True,
                key="premixing_pressure_plot_mode",
                help=(
                    "Actual Max Pressure plots left and right loadcell maxima. "
                    "Pressure Step plots the inferred 15-bar recipe steps."
                ),
            )
            batch_number_for_premix = batch.get("Batch no.") or "Batch"
            premix_figure = create_premixing_volume_step_plot(
                premix_summary,
                selected_premix_extruder,
                batch_number_for_premix,
                selected_premix_pressure_mode,
                selected_premixing_step_volume_ml,
            )
            apply_graph_style(premix_figure, graph_font_family, graph_font_size)
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

            p1, p2 = st.columns(2)
            p1.download_button(
                "Download Premixing Volume-Step Summary CSV",
                summary_view.to_csv(index=False).encode("utf-8"),
                f"{batch_number_for_premix}_extruder_{selected_premix_extruder}_premixing_summary.csv",
                "text/csv",
            )
            premix_html_key = (
                f"generated_html_premix_{selected_label}_{selected_premix_extruder}_"
                f"{selected_premix_pressure_mode}_{selected_premixing_step_volume_ml}"
            )
            if p2.button("Prepare Interactive Premixing HTML", key=f"prepare_{premix_html_key}"):
                with st.spinner("Preparing interactive Premixing HTML..."):
                    st.session_state[premix_html_key] = create_editable_interactive_html(
                        premix_figure, "Volume step"
                    )
            if premix_html_key in st.session_state:
                p2.download_button(
                    "Download Interactive Premixing Plot HTML",
                    st.session_state[premix_html_key],
                    f"{batch_number_for_premix}_extruder_{selected_premix_extruder}_premixing_plot.html",
                    "text/html",
                    key=f"download_{premix_html_key}",
                )

    with diagnostics_tab:
        st.markdown("#### Assigned process page ranges"); st.dataframe(process_ranges, use_container_width=True, hide_index=True)
        st.markdown("#### Page-level diagnostics"); st.dataframe(diagnostics, use_container_width=True, hide_index=True, height=400)
        st.markdown("#### Rejected rows")
        if rejected_rows.empty: st.success("No timestamp rows were rejected.")
        else: st.dataframe(rejected_rows, use_container_width=True, hide_index=True, height=400)

    batch_number = batch.get("Batch no.") or Path(
        selected_result["file_name"]
    ).stem
    safe_batch_number = safe_output_name(batch_number)
    selected_result_key = selector_labels[selected_label]

    st.markdown("### Export files")
    export_col1, export_col2 = st.columns(2)

    selected_excel_key = (
        f"generated_excel_{selected_result_key}_{selected_premixing_step_volume_ml}"
    )
    if export_col1.button(
        "Prepare Selected Batch Excel",
        key=f"prepare_{selected_excel_key}",
        use_container_width=True,
    ):
        with st.spinner("Preparing selected batch Excel workbook..."):
            st.session_state[selected_excel_key] = result_to_excel(selected_result)
    if selected_excel_key in st.session_state:
        export_col1.download_button(
            "⬇️ Download Selected Batch Excel",
            st.session_state[selected_excel_key],
            f"{safe_batch_number}_complete_process_extraction.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_{selected_excel_key}",
            type="primary",
            use_container_width=True,
        )

    uploaded_signature = "_".join(sorted(processed_results.keys()))
    all_batches_zip_key = (
        f"generated_zip_{hash(uploaded_signature)}_{premixing_step_volume_ml}"
    )
    if export_col2.button(
        "Prepare All Batch Excel Files",
        key=f"prepare_{all_batches_zip_key}",
        use_container_width=True,
    ):
        with st.spinner("Preparing Excel workbooks and ZIP package..."):
            st.session_state[all_batches_zip_key] = build_all_batches_zip(
                processed_results,
                extraction_errors,
            )
    if all_batches_zip_key in st.session_state:
        export_col2.download_button(
            "⬇️ Download All Batch Excel Files (ZIP)",
            st.session_state[all_batches_zip_key],
            "all_batch_process_extractions.zip",
            "application/zip",
            key=f"download_{all_batches_zip_key}",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
