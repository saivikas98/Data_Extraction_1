import re
import zipfile
import gc
import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import pymupdf as fitz
except ImportError:
    import fitz


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Batch Report Analyzer",
    layout="wide",
)

PUMP_COLOR_MAP = {
    "Pump 1": "#1f77b4",
    "Pump 2": "#ff7f0e",
    "Pump 3": "#2ca02c",
    "Pump 4": "#d62728",
}
FONT_OPTIONS = ["Times New Roman", "Arial", "Calibri", "Georgia", "Verdana"]


# =====================================================
# HELPERS
# =====================================================
def normalize_space(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def empty_dataframe(columns):
    return pd.DataFrame(columns=columns)


def find_first(text, patterns, default=""):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_space(match.group(1))
    return default


def find_section_bounds(pages, include_terms, stop_terms=None):
    """Return pages belonging to a report section based on page headers."""
    selected = []
    collecting = False
    stop_terms = stop_terms or []

    for page in pages:
        page_text = page["Text"]
        lower = page_text.lower()

        if any(term.lower() in lower for term in include_terms):
            collecting = True

        if collecting and selected and any(term.lower() in lower for term in stop_terms):
            break

        if collecting:
            selected.append(page)

    return selected


# =====================================================
# PDF TEXT EXTRACTION
# =====================================================
def extract_pdf_pages(pdf_bytes, parser_version="2026-10-26-v13-memory-safe"):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    raw_parts = []
    sorted_parts = []
    audit_section_active = False

    try:
        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text", sort=False) or ""
            sorted_text = page.get_text("text", sort=True) or ""
            raw_parts.append(raw_text)
            sorted_parts.append(sorted_text)
            lower_text = raw_text.lower()

            # Word coordinates are memory-heavy. Keep them only on possible
            # fill-weight pages where coordinate reconstruction may be needed.
            fill_candidate = any(
                marker in lower_text
                for marker in [
                    "nett weight", "net weight", "gross weight",
                    "tare weight", "weight passed",
                ]
            )
            words = page.get_text("words", sort=True) if fill_candidate else []

            audit_rows = []
            has_audit_header = (
                "userid" in lower_text
                and "description" in lower_text
                and "category" in lower_text
            )
            if "audit trail" in lower_text or has_audit_header:
                audit_section_active = True
            if audit_section_active and any(
                marker in lower_text
                for marker in [
                    "accumulator pressure", "heater temperature",
                    "load cell trend", "weight trend",
                ]
            ):
                audit_section_active = False

            if audit_section_active:
                try:
                    for table in page.find_tables().tables:
                        extracted = table.extract() or []
                        if not extracted:
                            continue
                        first_row = [normalize_space(cell) for cell in extracted[0]]
                        expected = {
                            "Timestamp", "State", "UserID",
                            "Description", "Comment", "Category",
                        }
                        if expected.issubset(set(first_row)):
                            audit_rows.extend(extracted[1:])
                        elif (
                            len(first_row) >= 6
                            and re.search(r"\d{2}/\d{2}/\d{4}", str(first_row[0]))
                        ):
                            audit_rows.extend(extracted)
                except Exception:
                    audit_rows = []

            # Sorted Text is not retained per page because sorted_full_text below
            # already contains it. This removes a large duplicate string copy.
            pages.append(
                {
                    "Page": page_number,
                    "Text": raw_text,
                    "Words": words,
                    "Audit Rows": audit_rows,
                }
            )
    finally:
        document.close()

    full_text = "\n".join(raw_parts)
    sorted_full_text = "\n".join(sorted_parts)
    del raw_parts, sorted_parts
    return full_text, sorted_full_text, pages


# =====================================================
# BATCH SUMMARY
# =====================================================
def extract_batch_summary(text):
    summary = {
        "Batch Number": find_first(
            text,
            [
                r"Batch\s*No\.?\s*:\s*(.*?)\s*Batch\s*Start\s*Time",
                r"Batch\s*Number\s*:\s*(.*?)\s*(?:\n|$)",
            ],
        ),
        "Batch Start": find_first(
            text,
            [r"Batch\s*Start\s*Time\s*:\s*(.*?)\s*Batch\s*End\s*Time"],
        ),
        "Batch End": find_first(
            text,
            [r"Batch\s*End\s*Time\s*:\s*(.*?)\s*Batch\s*Running\s*Time"],
        ),
        "Runtime": find_first(
            text,
            [r"Batch\s*Running\s*Time(?:\s*\([^)]*\))?\s*:\s*(.*?)\s*Batch\s*Size"],
        ),
        "Batch Size": find_first(
            text,
            [r"Batch\s*Size\s*:\s*(.*?)\s*(?:Product\s*Name|Recipe|\n)"],
        ),
        "Product Name": find_first(
            text,
            [r"Product\s*Name\s*:\s*(.*?)\s*(?:Batch\s*No|Recipe|\n)"],
        ),
        "Recipe Name": find_first(
            text,
            [
                r"Filling\s*Skid\s*Recipe\s*Name\s*:?\s*(.*?)\s*(?:\n|$)",
                r"Recipe\s*Name\s*:\s*(.*?)\s*(?:\n|$)",
            ],
        ),
    }
    return pd.DataFrame([summary])


# =====================================================
# PRODUCTION COUNTERS
# =====================================================
def extract_production_counters(text, sorted_text=""):
    columns = ["Counter Group", "Counter", "Value"]
    combined = f"{text}\n{sorted_text}"
    rows = []

    # The report stores labels first and their values below them, for example:
    # Infeed (pcs), Outfeed (pcs), Reject (pcs), followed by 1640, 1523, 117.
    section_patterns = [
        (
            "Production Counters",
            r"Production\s*Counters?\s*:\s*(.*?)\s*Check\s*Weigh\s*Reject\s*Counters?\s*:",
            ["Infeed (pcs)", "Outfeed (pcs)", "Reject (pcs)"],
        ),
        (
            "Check Weigh Reject Counters",
            r"Check\s*Weigh\s*Reject\s*Counters?\s*:\s*(.*?)\s*Air\s*Bubble\s*Reject\s*Counters?\s*:",
            ["Weigh Cell 1", "Weigh Cell 2", "Weigh Cell 3", "Weigh Cell 4", "Total"],
        ),
        (
            "Air Bubble Reject Counters",
            r"Air\s*Bubble\s*Reject\s*Counters?\s*:\s*(.*?)\s*Needle\s*Bonding\s*Reject\s*Counters?\s*:",
            ["Position 1", "Position 2", "Position 3", "Position 4", "Total"],
        ),
        (
            "Needle Bonding Reject Counters",
            r"Needle\s*Bonding\s*Reject\s*Counters?\s*:\s*(.*?)\s*Assembly\s*Reject\s*Counters?\s*:",
            ["Position 1", "Position 2", "Position 3", "Position 4", "Total"],
        ),
        (
            "Assembly Reject Counters",
            r"Assembly\s*Reject\s*Counters?\s*:\s*(.*?)\s*Filling\s*speed\s*:",
            ["Position 1", "Position 2", "Position 3", "Position 4", "Total"],
        ),
    ]

    for group, pattern, expected_labels in section_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        section = match.group(1)
        values = [int(x.replace(",", "")) for x in re.findall(r"(?m)^\s*(\d[\d,]*)\s*$", section)]
        if len(values) >= len(expected_labels):
            values = values[-len(expected_labels):]
            for label, value in zip(expected_labels, values):
                rows.append({"Counter Group": group, "Counter": label, "Value": value})

    # Direct inline fallback for other report versions.
    if not any(row["Counter Group"] == "Production Counters" for row in rows):
        inline = re.search(
            r"Infeed\s*(?:\(pcs\))?\s*[:=-]?\s*(\d+)\s+"
            r"Outfeed\s*(?:\(pcs\))?\s*[:=-]?\s*(\d+)\s+"
            r"Reject\s*(?:\(pcs\))?\s*[:=-]?\s*(\d+)",
            combined,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if inline:
            for label, value in zip(
                ["Infeed (pcs)", "Outfeed (pcs)", "Reject (pcs)"],
                inline.groups(),
            ):
                rows.append(
                    {
                        "Counter Group": "Production Counters",
                        "Counter": label,
                        "Value": int(value),
                    }
                )

    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["Counter Group", "Counter"], keep="first"
        ).reset_index(drop=True)
    return frame


def extract_filling_speed(text, sorted_text=""):
    combined = f"{text}\n{sorted_text}"
    match = re.search(
        r"Filling\s*speed\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*([^\n]*)",
        combined,
        flags=re.IGNORECASE,
    )
    if not match:
        return pd.DataFrame(columns=["Parameter", "Value", "Unit"])

    value = float(match.group(1).replace(",", "."))
    unit = normalize_space(match.group(2)) or "syringes/min"
    return pd.DataFrame(
        [{"Parameter": "Filling Speed", "Value": value, "Unit": unit}]
    )


# =====================================================
# FILL WEIGHT TABLE
# =====================================================
def _parse_fill_weight_stream(text):
    """Parse both standard and manual-assembly fill-weight row formats.

    Standard format:
        Date Time Nett Gross Tare Passed AirBubble NeedleBonding Assembly Position

    Manual-assembly format:
        Date Time Nett Gross Tare Passed Position
    """
    rows = []

    common_prefix = (
        r"(\d{2}/\d{2}/\d{4})\s+"
        r"(\d{2}:\d{2}:\d{2})\s+"
        r"(-?\d+(?:[.,]\d+)?)\s+"
        r"(-?\d+(?:[.,]\d+)?)\s+"
        r"(-?\d+(?:[.,]\d+)?)\s+"
        r"([01])\s+"
    )

    # Parse the longer standard format first so it cannot be partially consumed
    # by the shorter manual-assembly pattern.
    standard_pattern = re.compile(
        common_prefix
        + r"([01-])\s+"
        + r"([01-])\s+"
        + r"([01-])\s+"
        + r"([1-4])(?=\s|$)",
        flags=re.MULTILINE,
    )

    manual_pattern = re.compile(
        common_prefix
        + r"([1-4])(?=\s|$)",
        flags=re.MULTILINE,
    )

    occupied_spans = []

    def append_row(match, air_bubble="-", needle_bonding="-", assembly="-", position_group=7):
        values = [match.group(i).replace(",", ".") for i in (3, 4, 5)]
        rows.append(
            {
                "Timestamp": f"{match.group(1)} {match.group(2)}",
                "Nett Weight": float(values[0]),
                "Gross Weight": float(values[1]),
                "Tare Weight": float(values[2]),
                "Weight Passed": int(match.group(6)),
                "Air Bubble": air_bubble,
                "Needle Bonding": needle_bonding,
                "Assembly": assembly,
                "Position": int(match.group(position_group)),
            }
        )

    for match in standard_pattern.finditer(text):
        append_row(
            match,
            air_bubble=match.group(7),
            needle_bonding=match.group(8),
            assembly=match.group(9),
            position_group=10,
        )
        occupied_spans.append(match.span())

    def overlaps_standard(span):
        return any(span[0] < end and span[1] > start for start, end in occupied_spans)

    for match in manual_pattern.finditer(text):
        if overlaps_standard(match.span()):
            continue
        append_row(match, position_group=7)

    return rows


def _words_to_visual_lines(words, y_tolerance=2.5):
    """Rebuild visual rows from PyMuPDF words using y coordinates."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w[1] / y_tolerance), w[0]))
    lines = []
    current = []
    current_y = None
    for word in ordered:
        y = float(word[1])
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current.append(word)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            lines.append(" ".join(str(w[4]) for w in sorted(current, key=lambda x: x[0])))
            current = [word]
            current_y = y
    if current:
        lines.append(" ".join(str(w[4]) for w in sorted(current, key=lambda x: x[0])))
    return lines


def extract_fill_weights(text, sorted_text="", pages=None):
    columns = [
        "Timestamp", "Nett Weight", "Gross Weight", "Tare Weight",
        "Weight Passed", "Air Bubble", "Needle Bonding", "Assembly", "Position",
    ]
    rows = []

    # Try both native PDF reading orders first.
    rows.extend(_parse_fill_weight_stream(text))
    rows.extend(_parse_fill_weight_stream(sorted_text))

    # Reconstruct visual lines from word coordinates when plain text order is column-wise.
    if pages:
        for page in pages:
            lower = f"{page.get('Text', '')} {page.get('Sorted Text', '')}".lower()
            is_candidate = any(
                key in lower
                for key in ["nett weight", "net weight", "gross weight", "tare weight", "weight passed", "processed items"]
            )
            if not is_candidate:
                continue
            visual_text = "\n".join(_words_to_visual_lines(page.get("Words", [])))
            rows.extend(_parse_fill_weight_stream(visual_text))

    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], dayfirst=True, errors="coerce")
        frame = (
            frame.dropna(subset=["Timestamp"])
            .drop_duplicates()
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
    return frame


# =====================================================
# ALARM EXTRACTION
# =====================================================
def clean_alarm_chunk(text):
    """Remove repeating column headers and page footers before parsing alarms."""
    # The footer begins with Batch Number on every alarm page. Anything after that
    # is report metadata and must never become part of an alarm message.
    # Alarm data must end before the legend or Audit Trail, even when both
    # sections occur on the same PDF page as the final alarm record.
    text = re.split(
        r"(?im)^\s*(?:Alarm\s+State\s+Legend|Audit\s+Trail)\s*$",
        text,
        maxsplit=1,
    )[0]
    text = re.split(r"(?im)^\s*Batch\s+Number\s*:", text, maxsplit=1)[0]
    text = re.sub(r"(?im)^\s*(Timestamp|State|MsgText|Code)\s*$", "", text)
    text = re.sub(r"(?im)^\s*Page\s+\d+\s+of\s+\d+\s*$", "", text)
    return text.strip()


def parse_alarm_records(section_text, alarm_type):
    columns = ["Timestamp", "State", "Message", "Code", "Alarm Type"]
    rows = []
    section_text = clean_alarm_chunk(section_text)
    if not section_text:
        return pd.DataFrame(columns=columns)

    # A record is exactly: timestamp, state, message, integer code. The lookahead
    # constrains the message to end before the next timestamp or end of clean chunk.
    record_pattern = re.compile(
        r"(\d{2}/\d{2}/\d{4})\s+"
        r"(\d{2}:\d{2}:\d{2})\s+"
        r"([+*\-])\s+"
        r"(.+?)\s+"
        r"(\d{1,5})\s*"
        r"(?=(?:\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+[+*\-])|$)",
        flags=re.DOTALL,
    )

    for match in record_pattern.finditer(section_text):
        message = normalize_space(match.group(4))
        # Defensive removal of any metadata if a report variant puts footer text inline.
        message = re.split(
            r"\b(?:Batch Number|Product Name|Report Generated|Printed by|Page \d+ of \d+)\s*:",
            message,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        message = normalize_space(message)
        if not message:
            continue
        rows.append(
            {
                "Timestamp": f"{match.group(1)} {match.group(2)}",
                "State": match.group(3),
                "Message": message,
                "Code": match.group(5),
                "Alarm Type": alarm_type,
            }
        )

    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["Timestamp"] = pd.to_datetime(
            frame["Timestamp"], dayfirst=True, errors="coerce"
        )
        frame = (
            frame.dropna(subset=["Timestamp"])
            .drop_duplicates()
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
    return frame


def extract_alarm_tables(pages):
    """Parse Safety, GMP and Equipment sections page-by-page with header state."""
    category_aliases = [
        ("Safety", re.compile(r"(?im)^\s*(?:Alarms?\s+Safety|Safety\s+Alarms?)\s*$")),
        ("GMP", re.compile(r"(?im)^\s*(?:Alarms?\s+GMP|GMP\s+Alarms?)\s*$")),
        ("Equipment", re.compile(r"(?im)^\s*(?:Equipment\s+Alarms?|Alarms?\s+Equipment)\s*$")),
    ]
    chunks = {"Safety": [], "GMP": [], "Equipment": []}
    current_category = None

    for page in pages:
        page_text = page.get("Text", "") or ""
        # Remove the footer separately on every page. This fixes the observed case
        # where "Batch Number: ..." was appended to the final message on a page.
        page_text = re.split(
            r"(?im)^\s*Batch\s+Number\s*:", page_text, maxsplit=1
        )[0]

        # A page can contain the final alarm, the Alarm State Legend and the
        # beginning of Audit Trail. Truncate before the legend/audit header so
        # audit descriptions can never be appended to the final alarm message.
        boundary = re.search(
            r"(?im)^\s*(?:Alarm\s+State\s+Legend|Audit\s+Trail)\s*$",
            page_text,
        )
        reached_audit_boundary = boundary is not None
        if boundary:
            page_text = page_text[: boundary.start()]

        # A page can contain the end of one category and the beginning of another.
        header_hits = []
        for category, header_pattern in category_aliases:
            for hit in header_pattern.finditer(page_text):
                header_hits.append((hit.start(), hit.end(), category))
        header_hits.sort(key=lambda item: item[0])

        cursor = 0
        for hit_start, hit_end, new_category in header_hits:
            if current_category and hit_start > cursor:
                chunks[current_category].append(page_text[cursor:hit_start])
            current_category = new_category
            cursor = hit_end
        if current_category and cursor < len(page_text):
            chunks[current_category].append(page_text[cursor:])

        # Stop carrying alarm state after the legend/audit boundary.
        if reached_audit_boundary:
            current_category = None

    outputs = {}
    for category in ["Safety", "GMP", "Equipment"]:
        category_frames = []
        for chunk in chunks[category]:
            parsed = parse_alarm_records(chunk, category)
            if not parsed.empty:
                category_frames.append(parsed)
        if category_frames:
            outputs[category] = (
                pd.concat(category_frames, ignore_index=True)
                .drop_duplicates()
                .sort_values("Timestamp")
                .reset_index(drop=True)
            )
        else:
            outputs[category] = empty_dataframe(
                ["Timestamp", "State", "Message", "Code", "Alarm Type"]
            )
    return outputs["Safety"], outputs["GMP"], outputs["Equipment"]


def extract_safety_alarms(pages):
    return extract_alarm_tables(pages)[0]


def extract_gmp_alarms(pages):
    return extract_alarm_tables(pages)[1]


def extract_equipment_alarms(pages):
    return extract_alarm_tables(pages)[2]


# =====================================================
# AUDIT TRAIL EXTRACTION
# =====================================================
def extract_audit_trail(pages):
    columns = [
        "Page",
        "Timestamp",
        "State",
        "User",
        "Description",
        "Comment",
        "Category",
        "Action",
        "Audit Record",
    ]
    rows = []

    def clean_cell(value):
        value = str(value or "").replace("\n", " ")
        value = normalize_space(value)
        # Remove isolated layout underscores introduced by PDF cell extraction.
        value = re.sub(r"(?<!\w)_+(?!\w)", " ", value)
        return normalize_space(value)

    def refine_audit_text(value):
        """Repair words split by narrow PDF table cells without changing meaning."""
        value = clean_cell(value)
        if not value:
            return ""

        # Exact fragment repairs observed in this report and common HMI audit text.
        replacements = {
            r"\bN\s+ot\b": "Not",
            r"\bH\s+MI\b": "HMI",
            r"\bP\s+LC\b": "PLC",
            r"\bCon\s+nection\b": "Connection",
            r"\bConn\s+ection\b": "Connection",
            r"\bPowere\s+d\b": "Powered",
            r"\blogg\s+ed\b": "logged",
            r"\bM\s+anager\b": "Manager",
            r"\bv\s+alue\b": "value",
            r"\bBat\s+ch\b": "Batch",
            r"\bR\s+obot\b": "Robot",
            r"\bG\s+eneral\b": "General",
            r"\bU\s+nit\b": "Unit",
            r"\bG\s+uarding\b": "Guarding",
            r"\bD\s+oors\b": "Doors",
            r"\badministr\s+ation\b": "administration",
            r"\bAcknowledge\s+ment\b": "Acknowledgment",
            r"\bdescrip\s+tion\b": "description",
            r"\bproduc\s+tion\b": "production",
            r"\binter\s+face\b": "interface",
            r"\bInterface\s+PMQ\b": "Interface PMQ",
        }
        for pattern, replacement in replacements.items():
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

        # Repair known technical identifiers split by the narrow UserID column.
        value = re.sub(r"\bINF11DP\s+R-E(\d+)\b", r"INF11DPR-E\1", value)
        value = re.sub(r"/P\s+(\d{5,})\b", r"/P\1", value)
        value = re.sub(r"\bHMI[_ ]?Con\s*nection[_ ]?(\d+)\b", r"HMI_Connection_\1", value, flags=re.IGNORECASE)
        value = re.sub(r"\bHMI\s+Connection\s+(\d+)\b", r"HMI Connection \1", value, flags=re.IGNORECASE)

        # Repair a single trailing letter split from a sufficiently long word,
        # but only for endings commonly produced by column wrapping.
        value = re.sub(
            r"\b([A-Za-z]{4,})([ ]+)([dgry])\b",
            lambda m: m.group(1) + m.group(3)
            if (m.group(1) + m.group(3)).lower() in {
                "powered", "logged", "manager", "empty", "running",
                "stopped", "started", "changed", "closed", "opened",
            }
            else m.group(0),
            value,
            flags=re.IGNORECASE,
        )
        return normalize_space(value)

    def normalize_audit_timestamp(value):
        value = clean_cell(value)
        # The narrow timestamp column wraps the hour, e.g.
        # "02/08/2026 2\n2:32:01". After newline removal this becomes
        # "02/08/2026 2 2:32:01" and must be rejoined to 22:32:01.
        value = re.sub(
            r"(\d{2}/\d{2}/\d{4})\s+(\d)\s+(\d:\d{2}:\d{2})",
            r"\1 \2\3",
            value,
        )
        value = re.sub(
            r"(\d{2}/\d{2}/\d{4})\s+(\d{2})\s+(\d:\d{2}:\d{2})",
            r"\1 \2\3",
            value,
        )
        match = re.search(
            r"\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}",
            value,
        )
        return match.group(0) if match else ""

    for page in pages:
        page_number = page.get("Page", "")
        for raw_row in page.get("Audit Rows", []) or []:
            if not raw_row or len(raw_row) < 6:
                continue
            timestamp_text = normalize_audit_timestamp(raw_row[0])
            if not timestamp_text:
                continue

            state = clean_cell(raw_row[1])
            user = refine_audit_text(raw_row[2])
            description = refine_audit_text(raw_row[3])
            comment = refine_audit_text(raw_row[4])
            category = refine_audit_text(raw_row[5])

            # Category sometimes contains an alarm number such as "Alarm: 525".
            # Preserve it exactly rather than discarding the field.
            action = description
            audit_record = " | ".join(
                part
                for part in [state, user, description, comment, category]
                if part
            )
            rows.append(
                {
                    "Page": page_number,
                    "Timestamp": timestamp_text,
                    "State": state,
                    "User": user,
                    "Description": description,
                    "Comment": comment,
                    "Category": category,
                    "Action": action,
                    "Audit Record": audit_record,
                }
            )

    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["Timestamp"] = pd.to_datetime(
            frame["Timestamp"], dayfirst=True, errors="coerce"
        )
        frame = frame.dropna(subset=["Timestamp"]).reset_index(drop=True)
        # Do not drop duplicates. Some audit events genuinely repeat at the same
        # second and every printed table row must be retained.
    return frame


# =====================================================
# FINAL ALARM MESSAGE SANITIZATION
# =====================================================
def sanitize_alarm_messages(alarm_df):
    """Final guard against legend, audit-trail, and footer text entering alarms."""
    if alarm_df is None or alarm_df.empty:
        return alarm_df

    cleaned = alarm_df.copy()
    stop_pattern = re.compile(
        r"\s*(?:"
        r"Alarm\s+State\s+Legend|"
        r"Audit\s+Trail|"
        r"UserID\s+Description\s+Comment\s+Category|"
        r"Timestamp\s+State\s+UserID|"
        r"\+\s+Incoming\s+Alarm|"
        r"-\s+Outgoing\s+Alarm|"
        r"\*\s+Acknowledge\s+Alarm|"
        r"Batch\s+Number\s*:|"
        r"Product\s+Name\s*:|"
        r"Report\s+Generated\s*:|"
        r"Printed\s+by\s*:|"
        r"Page\s+\d+\s+of\s+\d+"
        r")",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def clean_message(value):
        message = normalize_space(value)
        return normalize_space(stop_pattern.split(message, maxsplit=1)[0])

    cleaned["Message"] = cleaned["Message"].map(clean_message)
    cleaned = cleaned[cleaned["Message"].str.len().gt(0)].copy()
    return cleaned.drop_duplicates().reset_index(drop=True)


# =====================================================
# FILL-WEIGHT STATISTICS
# =====================================================
def create_fill_weight_statistics(fill_pass_df, batch_label=""):
    columns = ["Batch", "Pump", "Sample Count", "Average", "Standard Deviation",
               "Relative Standard Deviation (%)", "Minimum", "Maximum", "Range"]
    if fill_pass_df is None or fill_pass_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    def add_row(label, series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        n = int(values.count())
        if not n:
            return
        avg = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else 0.0
        rows.append({"Batch": batch_label, "Pump": label, "Sample Count": n,
                     "Average": avg, "Standard Deviation": sd,
                     "Relative Standard Deviation (%)": (sd / avg * 100) if avg else 0.0,
                     "Minimum": float(values.min()), "Maximum": float(values.max()),
                     "Range": float(values.max() - values.min())})
    for position in sorted(fill_pass_df["Position"].dropna().unique().tolist()):
        add_row(f"Pump {int(position)}", fill_pass_df.loc[fill_pass_df["Position"].eq(position), "Nett Weight"])
    add_row("All Pumps Combined", fill_pass_df["Nett Weight"])
    out = pd.DataFrame(rows, columns=columns)
    for col in columns[3:]:
        out[col] = out[col].round(3)
    return out


def create_uploaded_batches_statistics(results):
    batch_tables = []
    all_passed = []
    for result in results:
        passed = result.get("fill_pass", pd.DataFrame())
        if passed.empty:
            continue
        batch_tables.append(create_fill_weight_statistics(passed, result["batch_label"]))
        all_passed.append(passed)
    batch_stats = pd.concat(batch_tables, ignore_index=True) if batch_tables else create_fill_weight_statistics(pd.DataFrame())
    combined_stats = create_fill_weight_statistics(
        pd.concat(all_passed, ignore_index=True) if all_passed else pd.DataFrame(),
        "All Uploaded Batches Combined",
    )
    return batch_stats, combined_stats


# =====================================================
# CROSS-BATCH STRENGTH COMPARISON
# =====================================================
def extract_strength_mg(summary_df):
    """Extract product strength in mg from Product Name / Recipe Name / Batch Size."""
    if summary_df is None or summary_df.empty:
        return None
    row = summary_df.iloc[0]
    searchable = " ".join(
        normalize_space(row.get(column, ""))
        for column in ["Product Name", "Recipe Name", "Batch Size"]
    )
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*mg\b", searchable)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def create_batch_pump_statistics_for_charts(results):
    """Create one row per batch/pump plus an all-pumps row for comparison charts."""
    rows = []
    for result in results:
        passed = result.get("fill_pass", pd.DataFrame())
        if passed is None or passed.empty:
            continue
        strength = extract_strength_mg(result.get("summary", pd.DataFrame()))
        if strength is None:
            continue
        batch = normalize_space(result.get("batch_label", ""))

        def append_statistics(pump_label, values):
            values = pd.to_numeric(values, errors="coerce").dropna()
            count = int(values.count())
            if count == 0:
                return
            mean_value = float(values.mean())
            sd_value = float(values.std(ddof=1)) if count > 1 else 0.0
            rsd_value = (sd_value / mean_value * 100.0) if mean_value else 0.0
            rows.append(
                {
                    "Batch": batch,
                    "Strength": strength,
                    "Pump": pump_label,
                    "Sample Count": count,
                    "Average": mean_value,
                    "Standard Deviation": sd_value,
                    "RSD (%)": rsd_value,
                }
            )

        for position in sorted(passed["Position"].dropna().unique().tolist()):
            append_statistics(
                f"Pump {int(position)}",
                passed.loc[passed["Position"].eq(position), "Nett Weight"],
            )
        append_statistics("All Pumps Combined", passed["Nett Weight"])

    return pd.DataFrame(rows)


def make_strength_comparison_chart(
    chart_df,
    metric,
    title,
    y_axis_title,
    font_family="Times New Roman",
    font_size=12,
):
    """Create interactive vertically stacked strength subplots."""
    if chart_df is None or chart_df.empty or metric not in chart_df.columns:
        return None

    strengths = sorted(chart_df["Strength"].dropna().unique().tolist(), reverse=True)
    if not strengths:
        return None

    pump_order = [
        "Pump 1", "Pump 2", "Pump 3", "Pump 4", "All Pumps Combined"
    ]
    color_map = {
        "Pump 1": "#0072BD",
        "Pump 2": "#E69F00",
        "Pump 3": "#009E73",
        "Pump 4": "#CC79A7",
        "All Pumps Combined": "#000000",
    }
    dash_map = {
        "Pump 1": "solid",
        "Pump 2": "solid",
        "Pump 3": "solid",
        "Pump 4": "solid",
        "All Pumps Combined": "dash",
    }
    symbol_map = {
        "Pump 1": "circle",
        "Pump 2": "circle",
        "Pump 3": "circle",
        "Pump 4": "circle",
        "All Pumps Combined": "diamond",
    }

    figure = make_subplots(
        rows=len(strengths),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10 if len(strengths) <= 3 else 0.07,
        subplot_titles=[f"{strength:g} mg" for strength in strengths],
    )

    legend_seen = set()
    for row_number, strength in enumerate(strengths, start=1):
        strength_df = chart_df[chart_df["Strength"].eq(strength)].copy()
        batch_order = strength_df["Batch"].drop_duplicates().tolist()
        for pump in pump_order:
            pump_df = strength_df[strength_df["Pump"].eq(pump)].copy()
            if pump_df.empty:
                continue
            pump_df["Batch"] = pd.Categorical(
                pump_df["Batch"], categories=batch_order, ordered=True
            )
            pump_df = pump_df.sort_values("Batch")
            show_legend = pump not in legend_seen
            legend_seen.add(pump)
            is_combined = pump == "All Pumps Combined"
            figure.add_trace(
                go.Scatter(
                    x=pump_df["Batch"].astype(str),
                    y=pump_df[metric],
                    mode="lines+markers+text" if is_combined else "lines+markers",
                    name=pump,
                    legendgroup=pump,
                    showlegend=show_legend,
                    line=dict(
                        color=color_map[pump],
                        width=2.2 if is_combined else 2.0,
                        dash=dash_map[pump],
                    ),
                    marker=dict(
                        color=color_map[pump],
                        size=8 if is_combined else 7,
                        symbol=symbol_map[pump],
                    ),
                    text=[f"{value:.2f}" for value in pump_df[metric]] if is_combined else None,
                    textposition="top center",
                    customdata=np.column_stack(
                        [pump_df["Sample Count"], pump_df["Strength"]]
                    ),
                    hovertemplate=(
                        "Batch: %{x}<br>"
                        + f"{y_axis_title}: %{{y:.3f}}<br>"
                        + "Samples: %{customdata[0]}<br>"
                        + "Strength: %{customdata[1]:g} mg<extra>"
                        + pump
                        + "</extra>"
                    ),
                ),
                row=row_number,
                col=1,
            )

    figure.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                family=font_family,
                size=max(int(font_size) + 8, 18),
                color="#000000",
            ),
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.04,
            yanchor="bottom",
            font=dict(family=font_family, size=font_size, color="#000000"),
        ),
        template="plotly_white",
        height=max(360 * len(strengths), 520),
        margin=dict(l=80, r=40, t=140, b=70),
        hovermode="closest",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family=font_family, size=font_size, color="#000000"),
    )

    for row_number in range(1, len(strengths) + 1):
        figure.update_xaxes(
            title_text="Batch Number" if row_number == len(strengths) else "",
            showgrid=True,
            gridcolor="#e0e0e0",
            linecolor="#000000",
            linewidth=1.5,
            mirror=True,
            tickfont=dict(family=font_family, size=font_size, color="#000000"),
            title_font=dict(family=font_family, size=font_size, color="#000000"),
            row=row_number,
            col=1,
        )
        figure.update_yaxes(
            title_text=y_axis_title,
            showgrid=True,
            gridcolor="#e0e0e0",
            linecolor="#000000",
            linewidth=1.5,
            mirror=True,
            tickfont=dict(family=font_family, size=font_size, color="#000000"),
            title_font=dict(family=font_family, size=font_size, color="#000000"),
            row=row_number,
            col=1,
        )

    for annotation in figure.layout.annotations:
        annotation.font = dict(
            family=font_family,
            size=max(int(font_size) + 2, 14),
            color="#000000",
        )
    return figure


# =====================================================
# SUMMARIES
# =====================================================
def create_alarm_summary(alarm_df):
    columns = ["Code", "Message", "Occurrences"]
    if alarm_df.empty:
        return empty_dataframe(columns)

    summary = (
        alarm_df.groupby(["Code", "Message"], dropna=False)
        .size()
        .reset_index(name="Occurrences")
        .sort_values(["Occurrences", "Message"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def create_audit_action_summary(audit_df):
    columns = ["Action", "Occurrences"]
    if audit_df.empty:
        return empty_dataframe(columns)

    summary = (
        audit_df.assign(Action=audit_df["Action"].fillna("").map(normalize_space))
        .query("Action != ''")
        .groupby("Action", dropna=False)
        .size()
        .reset_index(name="Occurrences")
        .sort_values(["Occurrences", "Action"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def create_audit_user_summary(audit_df):
    columns = ["User", "Occurrences"]
    if audit_df.empty or "User" not in audit_df.columns:
        return empty_dataframe(columns)

    user_data = audit_df[audit_df["User"].fillna("").str.strip().ne("")]
    if user_data.empty:
        return empty_dataframe(columns)

    return (
        user_data.groupby("User")
        .size()
        .reset_index(name="Occurrences")
        .sort_values("Occurrences", ascending=False)
        .reset_index(drop=True)
    )


# =====================================================
# PFS QUANTITY SUMMARY
# =====================================================
def get_pfs_quantity_summary(production_df, fill_df):
    """Return total filled, rejected, and accepted PFS quantities.

    Production counters are the primary source. Extracted fill-weight counts are
    used only as a fallback when production counters are unavailable.
    """
    total_filled = None
    total_accepted = None
    total_rejected = None

    if production_df is not None and not production_df.empty:
        counters = production_df[
            production_df["Counter Group"].eq("Production Counters")
        ].copy()
        if not counters.empty:
            counter_values = {
                normalize_space(row["Counter"]).lower(): int(row["Value"])
                for _, row in counters.iterrows()
            }
            total_filled = counter_values.get("infeed (pcs)")
            total_accepted = counter_values.get("outfeed (pcs)")
            total_rejected = counter_values.get("reject (pcs)")

    if fill_df is not None and not fill_df.empty:
        if total_filled is None:
            total_filled = int(len(fill_df))
        if total_accepted is None:
            total_accepted = int(fill_df["Weight Passed"].eq(1).sum())
        if total_rejected is None:
            total_rejected = int(fill_df["Weight Passed"].eq(0).sum())

    return {
        "Total Filled PFS": total_filled if total_filled is not None else 0,
        "Total Accepted PFS": total_accepted if total_accepted is not None else 0,
        "Total Rejected PFS": total_rejected if total_rejected is not None else 0,
    }


def add_pfs_quantity_annotations(fig, quantities, font_family, font_size):
    """Add one compact, bold PFS quantity summary below the graph title."""
    filled = int(quantities.get("Total Filled PFS", 0))
    accepted = int(quantities.get("Total Accepted PFS", 0))
    rejected = int(quantities.get("Total Rejected PFS", 0))

    summary_text = (
        f"<b>Total Filled PFS: {filled:,}</b>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<b>Total Accepted PFS: {accepted:,}</b>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<b>Total Rejected PFS: {rejected:,}</b>"
    )

    fig.add_annotation(
        x=0.5,
        y=1.095,
        xref="paper",
        yref="paper",
        text=summary_text,
        showarrow=False,
        align="center",
        xanchor="center",
        yanchor="middle",
        font=dict(
            family=font_family,
            size=max(int(font_size), 12),
            color="#000000",
        ),
    )
    fig.update_layout(margin=dict(l=70, r=40, t=170, b=75))
    return fig


# =====================================================
# PLOTLY FIGURES
# =====================================================
def apply_graph_style(fig, title, x_title, y_title, font_family="Times New Roman", font_size=12):
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        font=dict(family=font_family, size=font_size, color="#000000"),
        title_font=dict(
            family=font_family,
            size=max(font_size + 8, 18),
            color="#000000",
        ),
        legend_title_text="",
        legend=dict(font=dict(color="#000000")),
        hovermode="closest",
        margin=dict(l=60, r=30, t=80, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#b8b8b8",
        gridwidth=1,
        griddash="solid",
        showline=True,
        linecolor="#000000",
        linewidth=2,
        mirror=True,
        ticks="outside",
        tickcolor="#000000",
        tickfont=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        ),
        title_font=dict(family=font_family, size=font_size, color="#000000"),
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#b8b8b8",
        gridwidth=1,
        griddash="solid",
        showline=True,
        linecolor="#000000",
        linewidth=2,
        mirror=True,
        ticks="outside",
        tickcolor="#000000",
        tickfont=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        ),
        title_font=dict(family=font_family, size=font_size, color="#000000"),
        zeroline=False,
    )
    return fig


def make_fill_scatter(fill_pass_df, batch_label, lower_limit, upper_limit, font_family="Times New Roman", font_size=12, show_limits=True):
    plot_df = fill_pass_df.copy()
    plot_df["Pump"] = plot_df["Position"].map(lambda value: f"Pump {int(value)}")
    plot_df["Record Status"] = plot_df["Weight Passed"].map(
        {1: "Passed", 0: "Failed"}
    ).fillna("Unknown")
    fig = px.scatter(
        plot_df,
        x="Sample No",
        y="Nett Weight",
        color="Pump",
        symbol="Record Status",
        color_discrete_map=PUMP_COLOR_MAP,
        symbol_map={"Passed": "circle", "Failed": "x", "Unknown": "diamond"},
        category_orders={
            "Pump": ["Pump 1", "Pump 2", "Pump 3", "Pump 4"],
            "Record Status": ["Passed", "Failed", "Unknown"],
        },
        hover_data=[
            "Timestamp", "Gross Weight", "Tare Weight",
            "Weight Passed", "Position",
        ],
    )
    fig.update_traces(marker=dict(size=7, opacity=0.82))
    # Keep the pump/status legend in one horizontal row below the summary line.
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.035,
            yanchor="bottom",
            font=dict(
                family=font_family,
                size=font_size,
                color="#000000",
            ),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            itemsizing="constant",
            tracegroupgap=8,
        )
    )
    limit_shapes = [
        dict(
            type="line", xref="paper", x0=0, x1=1,
            yref="y", y0=upper_limit, y1=upper_limit,
            line=dict(color="#9e9e9e", width=4, dash="solid"),
        ),
        dict(
            type="line", xref="paper", x0=0, x1=1,
            yref="y", y0=lower_limit, y1=lower_limit,
            line=dict(color="#f28e2b", width=4, dash="solid"),
        ),
    ]
    limit_annotations = [
        dict(
            x=0.01, xref="paper", y=upper_limit, yref="y",
            text=f"Upper limit: {upper_limit:g}", showarrow=False,
            xanchor="left", yanchor="bottom",
            font=dict(color="#000000", family=font_family, size=font_size),
        ),
        dict(
            x=0.01, xref="paper", y=lower_limit, yref="y",
            text=f"Lower limit: {lower_limit:g}", showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(color="#000000", family=font_family, size=font_size),
        ),
    ]
    if show_limits:
        fig.update_layout(shapes=limit_shapes, annotations=limit_annotations)

    fig = apply_graph_style(
        fig,
        f"{batch_label} - Fill weight data",
        "Number of PFS",
        "Nett Weight",
        font_family,
        font_size,
    )
    # Reserve separate rows for title, PFS quantity summary, and horizontal legend.
    fig.update_layout(
        title=dict(
            text=f"{batch_label} - Fill weight data",
            x=0.5,
            xanchor="center",
            y=0.975,
            yanchor="top",
            pad=dict(t=0, b=0),
        ),
        margin=dict(l=70, r=40, t=170, b=75),
    )
    return fig


def make_fill_histogram(
    fill_data,
    batch_label,
    bin_width=5.0,
    font_family="Times New Roman",
    font_size=12,
):
    """Create a frequency chart with readable X-axis ranges such as 525-530 mg."""
    plot_df = fill_data.copy()
    weights = pd.to_numeric(plot_df["Nett Weight"], errors="coerce").dropna()

    if weights.empty:
        return apply_graph_style(
            px.bar(pd.DataFrame({"Weight Range": [], "Frequency": []})),
            f"{batch_label} - Fill weight data",
            "Nett Weight Range (mg)",
            "Frequency",
            font_family,
            font_size,
        )

    bin_width = max(float(bin_width), 0.1)
    minimum_edge = math.floor(float(weights.min()) / bin_width) * bin_width
    maximum_edge = math.ceil(float(weights.max()) / bin_width) * bin_width
    if maximum_edge <= minimum_edge:
        maximum_edge = minimum_edge + bin_width

    edges = np.arange(minimum_edge, maximum_edge + bin_width * 0.001, bin_width)
    if len(edges) < 2:
        edges = np.array([minimum_edge, minimum_edge + bin_width])

    def format_edge(value):
        return f"{value:.0f}" if float(value).is_integer() else f"{value:g}"

    labels = [
        f"{format_edge(edges[index])}-{format_edge(edges[index + 1])}"
        for index in range(len(edges) - 1)
    ]
    categories = pd.cut(
        weights,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    counts = categories.value_counts(sort=False).reindex(labels, fill_value=0)
    frequency_df = pd.DataFrame(
        {"Weight Range": labels, "Frequency": counts.astype(int).values}
    )

    fig = px.bar(
        frequency_df,
        x="Weight Range",
        y="Frequency",
        category_orders={"Weight Range": labels},
        color_discrete_sequence=["#5b9bd5"],
        text="Frequency",
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.8,
        textposition="outside",
        hovertemplate="Weight range: %{x} mg<br>Frequency: %{y}<extra></extra>",
    )
    fig = apply_graph_style(
        fig,
        f"{batch_label} - Fill weight data",
        "Nett Weight Range (mg)",
        "Frequency",
        font_family,
        font_size,
    )
    fig.update_xaxes(
        type="category",
        tickangle=0,
        tickfont=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        ),
        title_font=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        ),
    )
    fig.update_yaxes(
        tickfont=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        ),
        title_font=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        ),
    )
    fig.update_traces(
        textfont=dict(
            family=font_family,
            size=font_size,
            color="#000000",
        )
    )
    return fig


# =====================================================
# EXCEL EXPORT
# =====================================================
def build_excel(dataframes):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sheets_written = 0
        for sheet_name, dataframe in dataframes.items():
            if dataframe is None or dataframe.empty:
                continue
            export_df = dataframe.copy()
            for column in export_df.select_dtypes(include=["datetimetz"]).columns:
                export_df[column] = export_df[column].dt.tz_localize(None)
            export_df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )
            sheets_written += 1

        if sheets_written == 0:
            pd.DataFrame({"Information": ["No data was extracted."]}).to_excel(
                writer,
                sheet_name="Information",
                index=False,
            )

    output.seek(0)
    return output


# =====================================================
# INTERACTIVE PLOT CONFIGURATION
# =====================================================
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "displayModeBar": True,
    "editable": True,
    "edits": {
        "axisTitleText": True,
        "legendText": True,
        "annotationText": True,
        "titleText": True,
    },
    "toImageButtonOptions": {
        "format": "png",
        "filename": "interactive_graph",
        "height": 700,
        "width": 1400,
        "scale": 2,
    },
}


def make_interactive(fig, show_x_grid=True, show_y_grid=True, show_range_slider=False):
    """Apply stable Plotly interaction settings.

    The range-slider property is added only when enabled. Passing an explicit
    false range-slider update caused a Plotly/Streamlit rerender crash on some
    Community Cloud environments.
    """
    fig.update_layout(
        dragmode="zoom",
        hovermode="closest",
        clickmode="event+select",
        selectdirection="any",
        uirevision="keep-user-zoom",
    )

    xaxis_settings = dict(
        showgrid=show_x_grid,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
    )
    fig.update_xaxes(**xaxis_settings)

    # A newly generated figure has no range slider by default. Therefore, when
    # the user selects Off, do not send any range-slider property to Plotly.
    if bool(show_range_slider):
        fig.update_xaxes(rangeslider=dict(visible=True))

    fig.update_yaxes(
        showgrid=show_y_grid,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
    )
    return fig


def figure_to_html_bytes(fig):
    return fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config=PLOTLY_CONFIG,
    ).encode("utf-8")


def scatter_figure_to_html_bytes(fig, lower_limit, upper_limit, font_family, font_size, x_min=None, x_max=None, y_min=None, y_max=None):
    """Create standalone HTML with editable lower and upper limit inputs."""
    div_id = "fill_weight_scatter"
    chart_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", config=PLOTLY_CONFIG, div_id=div_id
    )
    safe_font = str(font_family).replace("\'", "")
    size = str(int(font_size))
    lower = repr(float(lower_limit))
    upper = repr(float(upper_limit))
    initial_x_min = "" if x_min is None else repr(float(x_min))
    initial_x_max = "" if x_max is None else repr(float(x_max))
    initial_y_min = "" if y_min is None else repr(float(y_min))
    initial_y_max = "" if y_max is None else repr(float(y_max))
    controls = (
        "<div style=\"font-family:" + safe_font + ";font-size:" + size + "px;padding:12px 16px;"
        "border:1px solid #d9d9d9;border-radius:8px;margin:10px 0;display:flex;gap:14px;"
        "align-items:end;flex-wrap:wrap;background:#fafafa;color:#000000;\">"
        "<label>Lower fill-weight limit<br><input id=\"lowerLimitInput\" type=\"number\" step=\"0.1\" value=\"" + lower + "\"></label>"
        "<label>Upper fill-weight limit<br><input id=\"upperLimitInput\" type=\"number\" step=\"0.1\" value=\"" + upper + "\"></label>"
        "<label>X-axis minimum<br><input id=\"xMinInput\" type=\"number\" step=\"1\" value=\"" + initial_x_min + "\"></label>"
        "<label>X-axis maximum<br><input id=\"xMaxInput\" type=\"number\" step=\"1\" value=\"" + initial_x_max + "\"></label>"
        "<label>Y-axis minimum<br><input id=\"yMinInput\" type=\"number\" step=\"0.1\" value=\"" + initial_y_min + "\"></label>"
        "<label>Y-axis maximum<br><input id=\"yMaxInput\" type=\"number\" step=\"0.1\" value=\"" + initial_y_max + "\"></label>"
        "<button id=\"applyLimitsButton\">Apply Limits</button>"
        "<button id=\"applyAxesButton\">Apply Axis Range</button>"
        "<button id=\"autoAxesButton\">Auto Axis Range</button>"
        "<button id=\"resetLimitsButton\">Reset Original Limits</button>"
        "<span id=\"limitMessage\" style=\"color:#b00020;\"></span></div>"
    )
    script = "".join([
        "<script>(function(){",
        "const plot=document.getElementById(\"" + div_id + "\");",
        "const lo=document.getElementById(\"lowerLimitInput\");",
        "const hi=document.getElementById(\"upperLimitInput\");",
        "const msg=document.getElementById(\"limitMessage\");",
        "const xMin=document.getElementById(\"xMinInput\");const xMax=document.getElementById(\"xMaxInput\");",
        "const yMin=document.getElementById(\"yMinInput\");const yMax=document.getElementById(\"yMaxInput\");",
        "const originalLower=" + lower + ";const originalUpper=" + upper + ";",
        "function layout(l,u){return {shapes:[",
        "{type:\"line\",xref:\"paper\",x0:0,x1:1,yref:\"y\",y0:u,y1:u,line:{color:\"#9e9e9e\",width:4}},",
        "{type:\"line\",xref:\"paper\",x0:0,x1:1,yref:\"y\",y0:l,y1:l,line:{color:\"#f28e2b\",width:4}}],",
        "annotations:[{x:0.01,xref:\"paper\",y:u,yref:\"y\",text:\"Upper limit: \"+u,showarrow:false},",
        "{x:0.01,xref:\"paper\",y:l,yref:\"y\",text:\"Lower limit: \"+l,showarrow:false}]};}",
        "function apply(){const l=Number(lo.value),u=Number(hi.value);",
        "if(!Number.isFinite(l)||!Number.isFinite(u)||l>=u){msg.textContent=\"Lower limit must be less than upper limit.\";return;}",
        "msg.textContent=\"\";Plotly.relayout(plot,layout(l,u));}",
        "function axisValue(el){return el.value.trim()===\"\"?null:Number(el.value);}",
        "function applyAxes(){const xmin=axisValue(xMin),xmax=axisValue(xMax),ymin=axisValue(yMin),ymax=axisValue(yMax);",
        "if((xmin!==null&&!Number.isFinite(xmin))||(xmax!==null&&!Number.isFinite(xmax))||(ymin!==null&&!Number.isFinite(ymin))||(ymax!==null&&!Number.isFinite(ymax))){msg.textContent=\"Axis limits must be valid numbers.\";return;}",
        "if(xmin!==null&&xmax!==null&&xmin>=xmax){msg.textContent=\"X-axis minimum must be less than maximum.\";return;}",
        "if(ymin!==null&&ymax!==null&&ymin>=ymax){msg.textContent=\"Y-axis minimum must be less than maximum.\";return;}",
        "msg.textContent=\"\";const update={};",
        "if(xmin!==null&&xmax!==null){update[\"xaxis.range\"]=[xmin,xmax];update[\"xaxis.autorange\"]=false;}else{update[\"xaxis.autorange\"]=true;}",
        "if(ymin!==null&&ymax!==null){update[\"yaxis.range\"]=[ymin,ymax];update[\"yaxis.autorange\"]=false;}else{update[\"yaxis.autorange\"]=true;}",
        "Plotly.relayout(plot,update);}",
        "document.getElementById(\"applyLimitsButton\").onclick=apply;",
        "document.getElementById(\"applyAxesButton\").onclick=applyAxes;",
        "document.getElementById(\"autoAxesButton\").onclick=function(){xMin.value=\"\";xMax.value=\"\";yMin.value=\"\";yMax.value=\"\";Plotly.relayout(plot,{\"xaxis.autorange\":true,\"yaxis.autorange\":true});};",
        "document.getElementById(\"resetLimitsButton\").onclick=function(){lo.value=originalLower;hi.value=originalUpper;apply();};",
        "lo.onkeydown=function(e){if(e.key===\"Enter\")apply();};hi.onkeydown=function(e){if(e.key===\"Enter\")apply();};",
        "})();</script>"
    ])
    control_css = (
        "<style>body{color:#000;background:#fff;}"
        "label{color:#000;font-weight:600;}"
        "input,button{color:#000;background:#fff;border:1.5px solid #000;"
        "border-radius:4px;padding:6px 10px;font-family:" + safe_font + ";"
        "font-size:" + size + "px;}"
        "button{cursor:pointer;font-weight:600;}"
        "</style>"
    )
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        + control_css
        + "</head><body>"
        + controls
        + chart_html
        + script
        + "</body></html>"
    )
    try:
        return html.encode("utf-8")
    except Exception:
        # Never allow an optional HTML export to terminate the Streamlit app.
        return figure_to_html_bytes(fig)

# =====================================================
# STREAMLIT UI
# =====================================================
# =====================================================
# MULTI-BATCH ANALYSIS
# =====================================================
def safe_file_stem(name):
    stem = str(name).rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "batch_report"


def analyze_batch_report(pdf_bytes, source_name):
    full_text, sorted_full_text, pages = extract_pdf_pages(pdf_bytes)
    summary_df = extract_batch_summary(full_text)
    production_df = extract_production_counters(full_text, sorted_full_text)
    fill_df = extract_fill_weights(full_text, sorted_full_text, pages)
    safety_df, gmp_df, equipment_df = extract_alarm_tables(pages)
    safety_df = sanitize_alarm_messages(safety_df)
    gmp_df = sanitize_alarm_messages(gmp_df)
    equipment_df = sanitize_alarm_messages(equipment_df)
    audit_df = extract_audit_trail(pages)

    fill_pass_df = fill_df[fill_df["Weight Passed"].eq(1)].copy()
    if not fill_pass_df.empty:
        fill_pass_df = fill_pass_df.sort_values("Timestamp").reset_index(drop=True)
        fill_pass_df["Sample No"] = range(1, len(fill_pass_df) + 1)

    alarm_frames = [df for df in [safety_df, gmp_df, equipment_df] if not df.empty]
    all_alarms_df = (
        pd.concat(alarm_frames, ignore_index=True)
        if alarm_frames
        else empty_dataframe(["Timestamp", "State", "Message", "Code", "Alarm Type"])
    )
    if not all_alarms_df.empty:
        all_alarms_df = all_alarms_df.sort_values("Timestamp").reset_index(drop=True)

    batch_number = ""
    if not summary_df.empty:
        batch_number = normalize_space(summary_df.iloc[0].get("Batch Number", ""))
    batch_label = batch_number or safe_file_stem(source_name)
    page_count = len(pages)

    # Retain the originating report for full traceability after multiple reports
    # belonging to the same manufacturing batch are combined.
    for frame in [fill_df, fill_pass_df, safety_df, gmp_df, equipment_df, all_alarms_df, audit_df]:
        if frame is not None and not frame.empty:
            frame["Source Report"] = source_name

    # Release PDF page text and word-coordinate structures before the result is
    # retained. Only structured DataFrames and the page count remain in memory.
    del full_text, sorted_full_text, pages
    gc.collect()

    return {
        "source_name": source_name,
        "batch_label": batch_label,
        "page_count": page_count,
        "summary": summary_df,
        "production": production_df,
        "fill": fill_df,
        "fill_pass": fill_pass_df,
        "safety": safety_df,
        "gmp": gmp_df,
        "equipment": equipment_df,
        "alarms": all_alarms_df,
        "alarm_summary": create_alarm_summary(all_alarms_df),
        "audit": audit_df,
        "audit_action_summary": create_audit_action_summary(audit_df),
        "audit_user_summary": create_audit_user_summary(audit_df),
    }


def report_root_batch(batch_label):
    """Return root batch, e.g. EG26051-1 -> EG26051."""
    value = normalize_space(batch_label)
    match = re.fullmatch(r"(.+?)-(\d+)", value)
    return match.group(1) if match else value


def combine_same_batch_reports(individual_results):
    """Combine report parts only when a root batch and/or multiple related parts exist."""
    labels = [normalize_space(result["batch_label"]) for result in individual_results]
    roots = {label: report_root_batch(label) for label in labels}
    root_counts = {}
    for root in roots.values():
        root_counts[root] = root_counts.get(root, 0) + 1

    grouped = {}
    for result in individual_results:
        label = normalize_space(result["batch_label"])
        root = roots[label]
        # A lone suffix batch remains separate. A root with two or more reports is merged.
        group_key = root if root_counts[root] > 1 else label
        grouped.setdefault(group_key, []).append(result)

    combined_results = []
    for group_key, parts in grouped.items():
        if len(parts) == 1:
            combined_results.append(parts[0])
            continue

        source_names = [part["source_name"] for part in parts]

        def combine_frames(key, sort_column=None, drop_duplicates=True):
            frames = [part[key] for part in parts if part.get(key) is not None and not part[key].empty]
            if not frames:
                return pd.DataFrame()
            frame = pd.concat(frames, ignore_index=True, sort=False)
            if drop_duplicates:
                frame = frame.drop_duplicates()
            if sort_column and sort_column in frame.columns:
                frame = frame.sort_values(sort_column, na_position="last")
            return frame.reset_index(drop=True)

        fill_df = combine_frames("fill", "Timestamp", True)
        fill_pass_df = fill_df[fill_df["Weight Passed"].eq(1)].copy() if not fill_df.empty else pd.DataFrame()
        if not fill_pass_df.empty:
            fill_pass_df = fill_pass_df.sort_values("Timestamp").reset_index(drop=True)
            fill_pass_df["Sample No"] = range(1, len(fill_pass_df) + 1)

        safety_df = combine_frames("safety", "Timestamp", True)
        gmp_df = combine_frames("gmp", "Timestamp", True)
        equipment_df = combine_frames("equipment", "Timestamp", True)
        alarms_df = combine_frames("alarms", "Timestamp", True)
        # Audit rows are not de-duplicated because repeated printed audit events can be genuine.
        audit_df = combine_frames("audit", "Timestamp", False)

        # Sum production counters from every report part.
        production_frames = [part["production"] for part in parts if not part["production"].empty]
        if production_frames:
            production_df = pd.concat(production_frames, ignore_index=True, sort=False)
            group_columns = [column for column in ["Counter Group", "Counter"] if column in production_df.columns]
            if group_columns and "Value" in production_df.columns:
                production_df = (
                    production_df.groupby(group_columns, dropna=False, as_index=False)["Value"]
                    .sum()
                )
        else:
            production_df = pd.DataFrame()

        # Consolidated summary: earliest start and latest end across report parts.
        summary_rows = [part["summary"].iloc[0].to_dict() for part in parts if not part["summary"].empty]
        consolidated = summary_rows[0].copy() if summary_rows else {"Batch Number": group_key}
        consolidated["Batch Number"] = group_key
        consolidated["Combined Report Count"] = len(parts)
        consolidated["Source Reports"] = "; ".join(source_names)

        starts = pd.to_datetime(
            [row.get("Batch Start", "") for row in summary_rows],
            dayfirst=True,
            errors="coerce",
        )
        ends = pd.to_datetime(
            [row.get("Batch End", "") for row in summary_rows],
            dayfirst=True,
            errors="coerce",
        )
        valid_starts = starts[~pd.isna(starts)]
        valid_ends = ends[~pd.isna(ends)]
        if len(valid_starts):
            consolidated["Batch Start"] = min(valid_starts).strftime("%d/%m/%Y %H:%M:%S")
        if len(valid_ends):
            consolidated["Batch End"] = max(valid_ends).strftime("%d/%m/%Y %H:%M:%S")
        if len(valid_starts) and len(valid_ends):
            duration = max(valid_ends) - min(valid_starts)
            total_seconds = int(duration.total_seconds())
            consolidated["Runtime"] = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
        summary_df = pd.DataFrame([consolidated])

        combined_results.append({
            "source_name": "; ".join(source_names),
            "source_reports": source_names,
            "report_count": len(parts),
            "batch_label": group_key,
            "page_count": sum(part.get("page_count", 0) for part in parts),
            "summary": summary_df,
            "production": production_df,
            "fill": fill_df,
            "fill_pass": fill_pass_df,
            "safety": safety_df,
            "gmp": gmp_df,
            "equipment": equipment_df,
            "alarms": alarms_df,
            "alarm_summary": create_alarm_summary(alarms_df),
            "audit": audit_df,
            "audit_action_summary": create_audit_action_summary(audit_df),
            "audit_user_summary": create_audit_user_summary(audit_df),
        })

    return sorted(combined_results, key=lambda item: item["batch_label"])


def render_batch_analysis(
    result, batch_key, lower_limit, upper_limit, histogram_bin_width,
    show_x_grid, show_y_grid, show_range_slider, marker_size,
    marker_opacity, font_family, font_size, show_fill_limits,
    custom_axis_range, x_axis_min, x_axis_max, y_axis_min, y_axis_max,
    all_results=None,
):
    """Render only the selected section to keep Streamlit Cloud memory stable."""
    batch_label = result["batch_label"]
    summary_df = result["summary"]
    production_df = result["production"]
    fill_df = result["fill"]
    fill_pass_df = result["fill_pass"]
    all_alarms_df = result["alarms"]
    alarm_summary_df = result["alarm_summary"]
    audit_df = result["audit"]
    audit_action_summary_df = result["audit_action_summary"]
    audit_user_summary_df = result["audit_user_summary"]
    fill_statistics_df = create_fill_weight_statistics(fill_pass_df, batch_label)
    pfs_quantities = get_pfs_quantity_summary(production_df, fill_df)

    st.subheader(batch_label)
    st.caption(f"Source report(s): {result['source_name']}")

    section = st.radio(
        "Analysis section",
        ["Summary", "Production", "Fill Weight", "Cross-Batch Statistics", "All Alarms", "Audit Trail", "Downloads"],
        horizontal=True,
        key=f"section_{batch_key}",
    )

    if section == "Summary":
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        total = len(fill_df)
        passed = len(fill_pass_df)
        failed = total - passed
        pass_rate = passed / total * 100 if total else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Fill Records", f"{total:,}")
        c2.metric("Fill Weight Passed", f"{passed:,}")
        c3.metric("Fill Weight Not Passed", f"{failed:,}")
        c4.metric("Pass Rate", f"{pass_rate:.2f}%")
        a1, a2, a3 = st.columns(3)
        a1.metric("All Alarm Events", f"{len(all_alarms_df):,}")
        a2.metric("Audit Records", f"{len(audit_df):,}")
        a3.metric("PDF Pages", f"{result.get('page_count', 0):,}")

    elif section == "Production":
        if production_df.empty:
            st.warning("No structured production counters were detected.")
        else:
            st.dataframe(production_df, use_container_width=True, hide_index=True)
            groups = production_df["Counter Group"].drop_duplicates().tolist()
            selected_group = st.selectbox(
                "Counter group to plot", groups, key=f"production_group_{batch_key}"
            )
            plot_df = production_df[production_df["Counter Group"].eq(selected_group)]
            fig = px.bar(plot_df, x="Counter", y="Value", color="Counter", text="Value")
            fig.update_traces(textposition="outside")
            fig = apply_graph_style(fig, selected_group, "Counter", "Value", font_family, font_size)
            fig = make_interactive(fig, show_x_grid, show_y_grid, False)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG,
                            key=f"production_plot_{batch_key}")

    elif section == "Fill Weight":
        if fill_df.empty:
            st.warning("No fill-weight records were detected.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Passed Records", f"{len(fill_pass_df):,}")
            f2.metric("Mean", f"{fill_pass_df['Nett Weight'].mean():.3f}" if not fill_pass_df.empty else "N/A")
            f3.metric("Standard Deviation", f"{fill_pass_df['Nett Weight'].std():.3f}" if len(fill_pass_df) > 1 else "N/A")
            f4.metric("Range", f"{fill_pass_df['Nett Weight'].min():.2f} to {fill_pass_df['Nett Weight'].max():.2f}" if not fill_pass_df.empty else "N/A")
            st.subheader("PFS Quantity Summary")
            q1, q2, q3 = st.columns(3)
            q1.metric("Total Filled PFS", f"{pfs_quantities['Total Filled PFS']:,}")
            q2.metric("Total Accepted PFS", f"{pfs_quantities['Total Accepted PFS']:,}")
            q3.metric("Total Rejected PFS", f"{pfs_quantities['Total Rejected PFS']:,}")

            st.subheader("Fill-Weight Statistical Summary by Pump")
            st.dataframe(fill_statistics_df, use_container_width=True, hide_index=True)
            record_scope = st.radio(
                "Fill-weight records for table and plots",
                ["Passed only", "All records"], horizontal=True,
                key=f"fill_record_scope_{batch_key}",
            )
            selected_fill_df = (
                fill_pass_df if record_scope == "Passed only" else fill_df
            ).sort_values("Timestamp").reset_index(drop=True).copy()
            selected_fill_df["Sample No"] = range(1, len(selected_fill_df) + 1)
            st.dataframe(selected_fill_df, use_container_width=True, hide_index=True)

            if not selected_fill_df.empty:
                m1, m2, m3 = st.columns(3)
                m1.metric("Plotted Records", f"{len(selected_fill_df):,}")
                m2.metric("Passed in Plot", f"{int(selected_fill_df['Weight Passed'].eq(1).sum()):,}")
                m3.metric("Failed in Plot", f"{int(selected_fill_df['Weight Passed'].eq(0).sum()):,}")
                positions = sorted(selected_fill_df["Position"].dropna().unique().tolist())
                selected_positions = st.multiselect(
                    "Pumps to display", positions, default=positions,
                    key=f"positions_{batch_key}",
                )
                graph_df = selected_fill_df[
                    selected_fill_df["Position"].isin(selected_positions)
                ].copy()
                if not graph_df.empty:
                    scatter_label = (
                        f"{batch_label} - Passed Records"
                        if record_scope == "Passed only"
                        else f"{batch_label} - All Records"
                    )
                    scatter = make_fill_scatter(
                        graph_df, scatter_label, lower_limit, upper_limit,
                        font_family, font_size, show_fill_limits,
                    )
                    scatter.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))
                    # Scatter plot gridlines are intentionally always retained.
                    scatter = make_interactive(scatter, True, True, show_range_slider)
                    scatter = add_pfs_quantity_annotations(
                        scatter,
                        pfs_quantities,
                        font_family,
                        font_size,
                    )
                    if custom_axis_range:
                        scatter.update_xaxes(range=[x_axis_min, x_axis_max], autorange=False)
                        scatter.update_yaxes(range=[y_axis_min, y_axis_max], autorange=False)
                    st.plotly_chart(scatter, use_container_width=True, config=PLOTLY_CONFIG,
                                    key=f"fill_scatter_{batch_key}")

                    histogram = make_fill_histogram(
                        graph_df, scatter_label, histogram_bin_width, font_family, font_size
                    )
                    histogram = make_interactive(histogram, show_x_grid, show_y_grid, False)
                    st.plotly_chart(histogram, use_container_width=True, config=PLOTLY_CONFIG,
                                    key=f"histogram_{batch_key}")

                    if st.button("Prepare Interactive Scatter HTML", key=f"prep_html_{batch_key}"):
                        html_bytes = scatter_figure_to_html_bytes(
                            scatter, lower_limit, upper_limit, font_family, font_size,
                            x_axis_min if custom_axis_range else None,
                            x_axis_max if custom_axis_range else None,
                            y_axis_min if custom_axis_range else None,
                            y_axis_max if custom_axis_range else None,
                        )
                        st.download_button(
                            "Download Interactive Scatter HTML", html_bytes,
                            file_name=f"{safe_file_stem(batch_label)}_fill_weight.html",
                            mime="text/html", key=f"html_download_{batch_key}",
                            use_container_width=True,
                        )

    elif section == "Cross-Batch Statistics":
        comparison_results = all_results if all_results is not None else [result]
        comparison_df = create_batch_pump_statistics_for_charts(comparison_results)
        if comparison_df.empty:
            st.warning(
                "Cross-batch graphs require passed fill-weight data and a product "
                "strength such as 60 mg, 90 mg, or 120 mg in the batch summary."
            )
        else:
            st.subheader("Cross-Batch Pump Statistics by Strength")
            st.caption(
                "Charts use records where Weight Passed = 1. "
                "All Pumps Combined is shown as a dashed black line with diamond markers."
            )
            selected_strengths = st.multiselect(
                "Strengths to display",
                options=sorted(
                    comparison_df["Strength"].dropna().unique().tolist(),
                    reverse=True,
                ),
                default=sorted(
                    comparison_df["Strength"].dropna().unique().tolist(),
                    reverse=True,
                ),
                key="cross_batch_strength_filter",
            )
            filtered_comparison_df = comparison_df[
                comparison_df["Strength"].isin(selected_strengths)
            ].copy()

            chart_choice = st.multiselect(
                "Graphs to display",
                options=["Mean", "Standard Deviation", "RSD (%)"],
                default=["Mean", "Standard Deviation", "RSD (%)"],
                key="cross_batch_chart_filter",
            )
            chart_specs = [
                (
                    "Mean",
                    "Average",
                    "Mean by Batch, Pump and Strength",
                    "Average Fill Weight (mg)",
                    "mean_by_batch_pump_strength",
                ),
                (
                    "Standard Deviation",
                    "Standard Deviation",
                    "Standard Deviation by Batch, Pump and Strength",
                    "Standard Deviation (mg)",
                    "sd_by_batch_pump_strength",
                ),
                (
                    "RSD (%)",
                    "RSD (%)",
                    "Relative Standard Deviation by Batch, Pump and Strength",
                    "RSD (%)",
                    "rsd_by_batch_pump_strength",
                ),
            ]

            for choice_label, metric, title, y_title, download_name in chart_specs:
                if choice_label not in chart_choice:
                    continue
                chart = make_strength_comparison_chart(
                    filtered_comparison_df,
                    metric,
                    title,
                    y_title,
                    font_family,
                    font_size,
                )
                if chart is None:
                    continue
                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key=f"{download_name}_{batch_key}",
                )
                st.download_button(
                    f"Download {choice_label} Chart as Interactive HTML",
                    data=figure_to_html_bytes(chart),
                    file_name=f"{download_name}.html",
                    mime="text/html",
                    key=f"download_{download_name}_{batch_key}",
                    use_container_width=True,
                )

            st.subheader("Statistics Used for Cross-Batch Charts")
            st.dataframe(
                filtered_comparison_df.sort_values(
                    ["Strength", "Batch", "Pump"],
                    ascending=[False, True, True],
                ),
                use_container_width=True,
                hide_index=True,
            )

    elif section == "All Alarms":
        if all_alarms_df.empty:
            st.warning("No structured alarm records were detected.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Alarm Events", f"{len(all_alarms_df):,}")
            c2.metric("Unique Alarm Types", f"{len(alarm_summary_df):,}")
            c3.metric("Repeated Alarm Types", f"{int((alarm_summary_df['Occurrences'] > 1).sum()):,}")
            st.dataframe(all_alarms_df, use_container_width=True, hide_index=True)
            st.subheader("Alarm Frequency Summary")
            st.dataframe(alarm_summary_df, use_container_width=True, hide_index=True)
            top = alarm_summary_df.head(20).sort_values("Occurrences")
            if not top.empty:
                fig = px.bar(top, x="Occurrences", y="Message", orientation="h")
                fig = apply_graph_style(fig, "Top Alarms", "Occurrences", "Alarm Message", font_family, font_size)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG,
                                key=f"alarm_plot_{batch_key}")

    elif section == "Audit Trail":
        if audit_df.empty:
            st.warning("No structured audit-trail records were detected.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Audit Records", f"{len(audit_df):,}")
            c2.metric("Unique Actions", f"{len(audit_action_summary_df):,}")
            c3.metric("Users", f"{len(audit_user_summary_df):,}")
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
            st.subheader("Audit Action Frequency Summary")
            st.dataframe(audit_action_summary_df, use_container_width=True, hide_index=True)

    elif section == "Downloads":
        st.caption("Excel is generated only when requested to avoid repeated memory spikes.")
        if st.button("Prepare Batch Excel Report", key=f"prep_excel_{batch_key}",
                     use_container_width=True):
            excel_file = build_excel({
                "Batch Summary": summary_df,
                "Production Counters": production_df,
                "Fill Weight All": fill_df,
                "Fill Weight Passed": fill_pass_df,
                "Fill Weight Statistics": fill_statistics_df,
                "PFS Quantity Summary": pd.DataFrame(
                    [{"Parameter": key, "Quantity": value} for key, value in pfs_quantities.items()]
                ),
                "All Alarms": all_alarms_df,
                "All Alarm Summary": alarm_summary_df,
                "Audit Trail": audit_df,
                "Audit Action Summary": audit_action_summary_df,
                "Audit User Summary": audit_user_summary_df,
            })
            st.download_button(
                "Download This Batch Excel Report", excel_file,
                file_name=f"{safe_file_stem(batch_label)}_Batch_Report_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_{batch_key}", use_container_width=True,
            )


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("Multi-Batch Report Analyzer")
st.caption("Upload multiple batch-report PDFs. Related report parts are merged by root batch number.")
st.info(
    "Memory-safe mode: PDF page text and coordinate data are released immediately "
    "after structured extraction. Streamlit data caching is disabled."
)
if st.sidebar.button("Clear App Data and Cache", type="primary", use_container_width=True):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    gc.collect()
    st.rerun()

uploaded_files = st.file_uploader(
    "Upload one or more batch report PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more PDF reports to begin.")
    st.stop()

st.sidebar.header("Graph Settings")
lower_limit = st.sidebar.number_input("Lower fill-weight limit", value=526.0)
upper_limit = st.sidebar.number_input("Upper fill-weight limit", value=540.0)
show_fill_limits = st.sidebar.checkbox("Show lower and upper limits in scatter plot and HTML export", value=True)
st.sidebar.subheader("Scatter Plot Axis Range")
custom_axis_range = st.sidebar.checkbox("Use custom X and Y axis limits", value=False)
x_axis_min = st.sidebar.number_input("X-axis minimum", value=1.0, disabled=not custom_axis_range)
x_axis_max = st.sidebar.number_input("X-axis maximum", value=1000.0, disabled=not custom_axis_range)
y_axis_min = st.sidebar.number_input("Y-axis minimum", value=520.0, disabled=not custom_axis_range)
y_axis_max = st.sidebar.number_input("Y-axis maximum", value=545.0, disabled=not custom_axis_range)
histogram_bin_width = st.sidebar.number_input(
    "Histogram weight interval (mg)",
    min_value=0.5,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Example: 5 mg displays ranges such as 525-530, 530-535, and 535-540.",
)
show_x_grid = st.sidebar.checkbox("Show X-axis gridlines", value=True)
show_y_grid = st.sidebar.checkbox("Show Y-axis gridlines", value=True)
range_slider_mode = st.sidebar.selectbox(
    "X-axis range slider",
    options=["Off", "On"],
    index=0,
    key="x_axis_range_slider_mode_v2",
)
show_range_slider = range_slider_mode == "On"
marker_size = st.sidebar.slider("Scatter marker size", 3, 16, 7)
marker_opacity = st.sidebar.slider("Scatter marker opacity", 0.2, 1.0, 0.82, 0.02)
st.sidebar.subheader("Graph Font Settings")
font_family = st.sidebar.selectbox("Font type", FONT_OPTIONS, index=0)
font_size = st.sidebar.number_input("Font size", min_value=8, max_value=48, value=12, step=1)

if lower_limit >= upper_limit:
    st.error("Lower fill-weight limit must be less than the upper limit.")
    st.stop()
if custom_axis_range and x_axis_min >= x_axis_max:
    st.error("X-axis minimum must be less than X-axis maximum.")
    st.stop()
if custom_axis_range and y_axis_min >= y_axis_max:
    st.error("Y-axis minimum must be less than Y-axis maximum.")
    st.stop()

analyze_clicked = st.button(
    "Analyze Uploaded Reports",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    # Remove any previous structured results before a new extraction run.
    st.session_state.pop("analysis_results", None)
    results = []
    progress = st.progress(0, text="Preparing batch analyses...")
    for index, uploaded in enumerate(uploaded_files):
        try:
            pdf_bytes = uploaded.getvalue()
            result = analyze_batch_report(pdf_bytes, uploaded.name)
            results.append(result)
            del pdf_bytes
            gc.collect()
        except Exception as exc:
            st.error(f"Could not process {uploaded.name}: {exc}")
        progress.progress(
            (index + 1) / len(uploaded_files),
            text=f"Processed {index + 1} of {len(uploaded_files)} reports",
        )
    progress.empty()

    if results:
        uploaded_report_count = len(results)
        results = combine_same_batch_reports(results)
        st.session_state["analysis_results"] = results
        st.session_state["uploaded_report_count"] = uploaded_report_count
    else:
        st.error("None of the uploaded reports could be processed.")

results = st.session_state.get("analysis_results", [])

if not results:
    st.info(
        "Files are selected but have not been processed. Click "
        "**Analyze Uploaded Reports** to start. The app will not automatically "
        "reprocess retained uploads when it starts or reruns."
    )
    st.stop()

uploaded_report_count = st.session_state.get("uploaded_report_count", len(results))
combined_batch_count = len(results)
merged_group_count = sum(
    1 for result in results if result.get("report_count", 1) > 1
)

st.success(
    f"Processed {uploaded_report_count} report(s) into "
    f"{combined_batch_count} batch analysis group(s). "
    f"Merged batch groups: {merged_group_count}."
)

try:
    batch_statistics_df, combined_statistics_df = create_uploaded_batches_statistics(results)
    st.header("Fill-Weight Statistical Summary for All Uploaded Batches")
    st.subheader("Batch-wise and Pump-wise Statistics")
    st.dataframe(batch_statistics_df, use_container_width=True, hide_index=True)
    st.subheader("All Uploaded Batches Combined")
    st.dataframe(combined_statistics_df, use_container_width=True, hide_index=True)
    if st.button("Prepare Combined Statistics Excel", use_container_width=True):
        combined_statistics_excel = build_excel({
            "Batch Pump Statistics": batch_statistics_df,
            "All Batches Combined": combined_statistics_df,
        })
        st.download_button(
            "Download Combined Fill-Weight Statistics",
            data=combined_statistics_excel,
            file_name="Combined_Fill_Weight_Statistics_All_Batches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
except Exception as statistics_error:
    st.warning(f"Combined statistics could not be displayed: {statistics_error}")

# Render only one batch at a time. Streamlit tabs execute all tab bodies and can
# keep every chart in memory, which is unsafe for multiple 300-500 page reports.
batch_labels = [result["batch_label"] for result in results]
selected_batch_label = st.selectbox("Batch analysis to display", batch_labels)
selected_index = batch_labels.index(selected_batch_label)
selected_result = results[selected_index]
batch_key = f"batch_{selected_index + 1}_{safe_file_stem(selected_result['source_name'])}"

try:
    render_batch_analysis(
        selected_result,
        batch_key,
        lower_limit,
        upper_limit,
        histogram_bin_width,
        show_x_grid,
        show_y_grid,
        show_range_slider,
        marker_size,
        marker_opacity,
        font_family,
        int(font_size),
        show_fill_limits,
        custom_axis_range,
        x_axis_min,
        x_axis_max,
        y_axis_min,
        y_axis_max,
        results,
    )
except Exception as display_error:
    st.error("The selected batch dashboard could not be rendered.")
    st.exception(display_error)
    st.info("Use Clear App Data and Cache, upload the report again, and click Analyze Uploaded Reports.")

# render_batch_analysis already provides the selected batch workbook. Do not build
# every batch workbook and a ZIP automatically on each widget rerun.

with st.expander("Multi-batch extraction summary"):
    summary_rows = []
    for result in results:
        summary_rows.append({
            "Source Report(s)": result["source_name"],
            "Reports Combined": result.get("report_count", 1),
            "Batch": result["batch_label"],
            "Fill Records": len(result["fill"]),
            "Passed Fill Records": len(result["fill_pass"]),
            "Alarm Events": len(result["alarms"]),
            "Audit Records": len(result["audit"]),
            "PDF Pages": result.get("page_count", 0),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption("Parser version: 2026-10-30-v32-cross-batch-strength-charts")
