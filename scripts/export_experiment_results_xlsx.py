#!/usr/bin/env python3
"""Export the SD-LoRA experiment inventory to a self-contained XLSX workbook.

The workbook keeps physical logs and the previously imported historical ledger
separate, then provides a cross-source deduplicated view for analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("", ROOT_NS)
ET.register_namespace("r", REL_NS)


def load_parser(adaptive_root: Path):
    sys.path.insert(0, str(adaptive_root))
    from scripts.export_experiment_ledger import parse_log, signature

    return parse_log, signature


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def enrich(row: dict[str, str], record_source: str) -> dict[str, str]:
    result = dict(row)
    result["record_source"] = record_source
    result.setdefault("source", record_source)
    result.setdefault("repo", "")
    result.setdefault("log_path", "")
    result.setdefault("config_path", "")
    result.setdefault("dataset", "")
    result.setdefault("method", "")
    result.setdefault("num_tasks", "")
    result.setdefault("final_top1", "")
    result.setdefault("aaa", "")
    result.setdefault("forgetting", "")
    result.setdefault("top1_curve", "")
    result.setdefault("top5_curve", "")
    result.setdefault("modified_epoch", "")
    result.setdefault("duplicate_count", "1")
    result.setdefault("all_log_paths", result.get("log_path", ""))
    return result


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row.get(key, "")
        for key in (
            "dataset",
            "method",
            "num_tasks",
            "final_top1",
            "aaa",
            "forgetting",
            "top1_curve",
        )
    )


def deduplicate(rows: list[dict[str, str]], signature_fn=None) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row_key(row)].append(row)

    result = []
    for entries in groups.values():
        canonical = dict(min(entries, key=lambda item: item.get("log_path", "")))
        paths = []
        origins = []
        for item in entries:
            for raw in item.get("all_log_paths", "").split(" | "):
                if raw and raw not in paths:
                    paths.append(raw)
            origin = item.get("record_source", item.get("source", ""))
            if origin and origin not in origins:
                origins.append(origin)
        canonical["duplicate_count"] = str(len(paths) or len(entries))
        canonical["all_log_paths"] = " | ".join(paths)
        canonical["origin_count"] = str(len(entries))
        canonical["origin_sources"] = " | ".join(origins)
        result.append(canonical)
    return sorted(result, key=lambda item: (item.get("dataset", ""), item.get("method", ""), item.get("log_path", "")))


def scan_physical_logs(project_root: Path, parse_log) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parsed = []
    unparsed = []
    for path in sorted(project_root.rglob("*.log")):
        row = parse_log(path, "physical", project_root)
        if row is None:
            try:
                stat = path.stat()
                unparsed.append(
                    {
                        "log_path": str(path.relative_to(project_root)),
                        "size_bytes": str(stat.st_size),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                        "tail": path.read_text(errors="replace")[-500:].replace("\n", " "),
                    }
                )
            except OSError:
                pass
            continue
        parsed.append(enrich(row, "physical"))
    return parsed, unparsed


def as_float(value: str):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def aggregate(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    result = []
    for key_values, entries in sorted(groups.items()):
        output = {key: value for key, value in zip(keys, key_values)}
        output["records"] = str(len(entries))
        for metric in ("final_top1", "aaa", "forgetting"):
            values = [as_float(item.get(metric, "")) for item in entries]
            values = [value for value in values if value is not None]
            output[f"{metric}_mean"] = f"{sum(values) / len(values):.6f}" if values else ""
            output[f"{metric}_min"] = f"{min(values):.6f}" if values else ""
            output[f"{metric}_max"] = f"{max(values):.6f}" if values else ""
        result.append(output)
    return result


def excel_col(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, rem = divmod(value - 1, 26)
        result = chr(65 + rem) + result
    return result


def cell_value(value):
    if value is None:
        return ""
    text = str(value)
    # XML 1.0 rejects most ASCII control characters that can occur in log tails.
    return "".join(
        char
        for char in text
        if char in "\t\n\r"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
    )


def make_styles() -> bytes:
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{ROOT_NS}">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts>
  <fonts count="2">
    <font><sz val="10"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="1" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    return xml.encode("utf-8")


def make_sheet(rows: list[dict[str, str]], headers: list[str]) -> bytes:
    root = ET.Element(f"{{{ROOT_NS}}}worksheet")
    views = ET.SubElement(root, f"{{{ROOT_NS}}}sheetViews")
    view = ET.SubElement(views, f"{{{ROOT_NS}}}sheetView", {"workbookViewId": "0"})
    ET.SubElement(view, f"{{{ROOT_NS}}}pane", {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"})
    ET.SubElement(root, f"{{{ROOT_NS}}}sheetFormatPr", {"defaultRowHeight": "15"})
    widths = []
    for header in headers:
        widths.append(max(12, min(55, len(header) + 2)))
    for row in rows:
        for index, header in enumerate(headers):
            widths[index] = max(widths[index], min(55, max((len(line) for line in cell_value(row.get(header, "")).split("\n")), default=0) + 2))
    cols = ET.SubElement(root, f"{{{ROOT_NS}}}cols")
    for index, width in enumerate(widths):
        ET.SubElement(cols, f"{{{ROOT_NS}}}col", {"min": str(index + 1), "max": str(index + 1), "width": f"{width:.1f}", "customWidth": "1"})
    sheet_data = ET.SubElement(root, f"{{{ROOT_NS}}}sheetData")
    numeric_int = {"records", "num_tasks", "duplicate_count", "origin_count", "size_bytes"}
    numeric_decimal = {"final_top1", "aaa", "forgetting"}
    for row_number, row in enumerate([dict(zip(headers, headers))] + rows, start=1):
        row_node = ET.SubElement(sheet_data, f"{{{ROOT_NS}}}row", {"r": str(row_number)})
        for index, header in enumerate(headers):
            raw = row.get(header, "")
            ref = f"{excel_col(index)}{row_number}"
            if row_number == 1:
                cell = ET.SubElement(row_node, f"{{{ROOT_NS}}}c", {"r": ref, "t": "inlineStr", "s": "1"})
                inline = ET.SubElement(cell, f"{{{ROOT_NS}}}is")
                ET.SubElement(inline, f"{{{ROOT_NS}}}t").text = cell_value(raw)
                continue
            if header in numeric_int and as_float(raw) is not None:
                cell = ET.SubElement(row_node, f"{{{ROOT_NS}}}c", {"r": ref, "s": "2"})
                ET.SubElement(cell, f"{{{ROOT_NS}}}v").text = str(int(float(raw)))
            elif header in numeric_decimal and as_float(raw) is not None:
                cell = ET.SubElement(row_node, f"{{{ROOT_NS}}}c", {"r": ref, "s": "3"})
                ET.SubElement(cell, f"{{{ROOT_NS}}}v").text = f"{float(raw):.9f}"
            else:
                cell = ET.SubElement(row_node, f"{{{ROOT_NS}}}c", {"r": ref, "t": "inlineStr"})
                inline = ET.SubElement(cell, f"{{{ROOT_NS}}}is")
                text = ET.SubElement(inline, f"{{{ROOT_NS}}}t")
                text.text = cell_value(raw)
    if headers:
        end = f"{excel_col(len(headers) - 1)}{len(rows) + 1}"
        ET.SubElement(root, f"{{{ROOT_NS}}}autoFilter", {"ref": f"A1:{end}"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_xlsx(path: Path, sheets: list[tuple[str, list[dict[str, str]], list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = ET.Element(f"{{{CONTENT_TYPES_NS}}}Types")
    ET.SubElement(content_types, f"{{{CONTENT_TYPES_NS}}}Default", {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, f"{{{CONTENT_TYPES_NS}}}Default", {"Extension": "xml", "ContentType": "application/xml"})
    ET.SubElement(content_types, f"{{{CONTENT_TYPES_NS}}}Override", {"PartName": "/xl/workbook.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"})
    ET.SubElement(content_types, f"{{{CONTENT_TYPES_NS}}}Override", {"PartName": "/xl/styles.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"})
    for index in range(len(sheets)):
        ET.SubElement(content_types, f"{{{CONTENT_TYPES_NS}}}Override", {"PartName": f"/xl/worksheets/sheet{index + 1}.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"})

    workbook = ET.Element(f"{{{ROOT_NS}}}workbook")
    sheets_node = ET.SubElement(workbook, f"{{{ROOT_NS}}}sheets")
    for index, (name, _, _) in enumerate(sheets, start=1):
        ET.SubElement(sheets_node, f"{{{ROOT_NS}}}sheet", {"name": name[:31], "sheetId": str(index), f"{{{REL_NS}}}id": f"rId{index + 1}"})

    rels = ET.Element(f"{{{PKG_REL_NS}}}Relationships")
    ET.SubElement(rels, f"{{{PKG_REL_NS}}}Relationship", {"Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "Target": "xl/workbook.xml"})
    workbook_rels = ET.Element(f"{{{PKG_REL_NS}}}Relationships")
    ET.SubElement(workbook_rels, f"{{{PKG_REL_NS}}}Relationship", {"Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "Target": "styles.xml"})
    for index in range(len(sheets)):
        ET.SubElement(workbook_rels, f"{{{PKG_REL_NS}}}Relationship", {"Id": f"rId{index + 2}", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet", "Target": f"worksheets/sheet{index + 1}.xml"})

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
        archive.writestr("_rels/.rels", ET.tostring(rels, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/workbook.xml", ET.tostring(workbook, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(workbook_rels, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/styles.xml", make_styles())
        for index, (_, rows, headers) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", make_sheet(rows, headers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-ledger", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    adaptive_root = args.adaptive_root.resolve()
    parse_log, _ = load_parser(adaptive_root)

    physical_raw, unparsed = scan_physical_logs(project_root, parse_log)
    physical = deduplicate(physical_raw)
    historical_raw = [enrich(row, "historical_ledger") for row in read_csv_rows(args.historical_ledger)]
    historical = deduplicate(historical_raw)
    combined = deduplicate(physical + historical)

    latest = [row for row in physical if "20260918_143900" in row.get("log_path", "")]
    latest = sorted(latest, key=lambda row: (row.get("dataset", ""), int(row.get("num_tasks", "0") or 0)))

    summary_rows = [
        {"metric": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"metric": "physical_log_files", "value": str(len(list(project_root.rglob("*.log"))))},
        {"metric": "physical_completed_log_records", "value": str(len(physical_raw))},
        {"metric": "physical_deduplicated_records", "value": str(len(physical))},
        {"metric": "physical_unparsed_logs", "value": str(len(unparsed))},
        {"metric": "historical_ledger_rows", "value": str(len(historical_raw))},
        {"metric": "historical_deduplicated_records", "value": str(len(historical))},
        {"metric": "combined_deduplicated_records", "value": str(len(combined))},
        {"metric": "combined_datasets", "value": ", ".join(sorted({row.get("dataset", "") for row in combined if row.get("dataset")}))},
        {"metric": "latest_tasklen_records", "value": str(len(latest))},
    ]

    record_headers = [
        "record_source", "source", "repo", "log_path", "config_path", "dataset", "method", "num_tasks",
        "final_top1", "aaa", "forgetting", "top1_curve", "top5_curve", "modified_epoch", "duplicate_count",
        "origin_count", "origin_sources", "all_log_paths",
    ]
    dataset_summary = aggregate(combined, ("dataset",))
    method_summary = aggregate(combined, ("dataset", "method", "num_tasks"))
    aggregate_headers = list(dataset_summary[0].keys()) if dataset_summary else ["dataset", "records"]
    method_headers = list(method_summary[0].keys()) if method_summary else ["dataset", "method", "num_tasks", "records"]

    sheets = [
        ("Summary", summary_rows, ["metric", "value"]),
        ("All Records", combined, record_headers),
        ("Physical Records", physical, record_headers),
        ("Historical Ledger", historical, record_headers),
        ("Latest TaskLen", latest, record_headers),
        ("Dataset Summary", dataset_summary, aggregate_headers),
        ("Method Summary", method_summary, method_headers),
        ("Unparsed Logs", unparsed, ["log_path", "size_bytes", "modified", "tail"]),
    ]
    write_xlsx(args.output.resolve(), sheets)
    print(
        "generated={} physical_raw={} physical_unique={} historical={} combined={} latest={} unparsed={} output={}".format(
            args.output.resolve(), len(physical_raw), len(physical), len(historical), len(combined), len(latest), len(unparsed), args.output.resolve()
        )
    )


if __name__ == "__main__":
    main()
