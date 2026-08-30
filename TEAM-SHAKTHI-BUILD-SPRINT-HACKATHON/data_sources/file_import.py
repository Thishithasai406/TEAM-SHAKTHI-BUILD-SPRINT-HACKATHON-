import csv
import io
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from data_sources.base import DealDataProvider
from data_sources.normalizer import normalize_deal

COLUMN_MAPPINGS = {
    "deal_id": ["deal_id", "deal id", "opportunity id", "id", "opportunity_id", "opportunity"],
    "company": ["company", "company name", "account", "account name", "customer"],
    "deal_size": ["deal_size", "amount", "deal size", "deal value", "value", "opportunity amount"],
    "stage": ["stage", "deal stage", "opportunity stage", "stage name"],
    "days_in_stage": ["days_in_stage", "days in stage", "stage duration", "duration"],
    "response_latency_hrs": ["response_latency_hrs", "response latency", "latency", "buyer latency"],
    "sentiment_trend": ["sentiment_trend", "sentiment", "communication sentiment"],
    "stakeholder_count": ["stakeholder_count", "stakeholders", "stakeholder count", "contacts involved"],
    "competitor_mentions": ["competitor_mentions", "competitors", "competitor count", "competitor mentions"],
    "scope_change_flags": ["scope_change_flags", "scope changes", "scope flags", "custom scope flags"]
}

class FileImportDataProvider(DealDataProvider):
    def __init__(self):
        self.imported_deals: List[Dict[str, Any]] = []
        self.last_imported_at: Optional[str] = None
        self.import_summary: Dict[str, Any] = {"valid_rows": 0, "skipped_rows": 0, "errors": []}

    def process_file_content(self, filename: str, content: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parses CSV or Excel bytes, validates required fields, normalizes rows into canonical schema."""
        rows_data = []
        errors = []
        
        # 1. Parse CSV or Excel
        if filename.endswith(".csv"):
            try:
                text_stream = io.StringIO(content.decode("utf-8-sig", errors="ignore"))
                reader = csv.DictReader(text_stream)
                fieldnames = reader.fieldnames or []
                rows_data = list(reader)
            except Exception as e:
                return [], {"valid_rows": 0, "skipped_rows": 0, "errors": [f"CSV Read Error: {str(e)}"]}
        elif filename.endswith((".xlsx", ".xls")):
            try:
                import pandas as pd
                excel_file = io.BytesIO(content)
                df = pd.read_excel(excel_file)
                df = df.where(pd.notnull(df), None)
                fieldnames = list(df.columns)
                rows_data = df.to_dict(orient="records")
            except Exception as e:
                return [], {"valid_rows": 0, "skipped_rows": 0, "errors": [f"Excel Read Error (pandas/openpyxl required): {str(e)}"]}
        else:
            return [], {"valid_rows": 0, "skipped_rows": 0, "errors": ["Unsupported file type. Please upload .csv, .xlsx, or .xls"]}

        if not fieldnames or not rows_data:
            return [], {"valid_rows": 0, "skipped_rows": 0, "errors": ["File is empty or missing header row"]}

        # 2. Header Mapping
        normalized_headers = {str(f).strip().lower(): str(f) for f in fieldnames}
        header_map = {}
        for target_field, aliases in COLUMN_MAPPINGS.items():
            for alias in aliases:
                if alias in normalized_headers:
                    header_map[target_field] = normalized_headers[alias]
                    break

        if "deal_id" not in header_map and "company" not in header_map:
            return [], {"valid_rows": 0, "skipped_rows": len(rows_data), "errors": ["Missing required identifier column: deal_id or company"]}

        valid_deals = []
        skipped_count = 0
        now_str = datetime.datetime.utcnow().isoformat()

        for idx, row in enumerate(rows_data, start=2):  # row 1 is header
            mapped_row = {}
            for target_field, orig_col in header_map.items():
                val = row.get(orig_col)
                mapped_row[target_field] = val

            deal_id = mapped_row.get("deal_id") or f"FILE-ROW-{idx}"
            company = mapped_row.get("company") or f"Imported Account {idx}"

            # Basic Validation
            raw_size = mapped_row.get("deal_size")
            if raw_size is not None:
                try:
                    mapped_row["deal_size"] = float(str(raw_size).replace("$", "").replace(",", "").strip())
                except ValueError:
                    errors.append(f"Row {idx}: Invalid deal_size '{raw_size}', defaulted to 0")
                    mapped_row["deal_size"] = 0.0

            mapped_row["deal_id"] = str(deal_id).strip()
            mapped_row["company"] = str(company).strip()
            mapped_row["source"] = "file"
            mapped_row["source_record_id"] = str(deal_id).strip()
            mapped_row["created_at"] = now_str
            mapped_row["updated_at"] = now_str

            normalized = normalize_deal(mapped_row, source_type="file")
            valid_deals.append(normalized)

        self.imported_deals = valid_deals
        self.last_imported_at = now_str
        self.import_summary = {
            "valid_rows": len(valid_deals),
            "skipped_rows": skipped_count,
            "errors": errors
        }
        return valid_deals, self.import_summary

    def get_deals(self) -> List[Dict[str, Any]]:
        return self.imported_deals

    def get_deal(self, deal_id: str) -> Optional[Dict[str, Any]]:
        for d in self.imported_deals:
            if d.get("deal_id") == deal_id or d.get("source_record_id") == deal_id:
                return d
        return None

    def health_check(self) -> Dict[str, Any]:
        return {
            "source": "file",
            "connected": len(self.imported_deals) > 0,
            "status": f"Connected ({len(self.imported_deals)} deals imported)" if self.imported_deals else "No file uploaded",
            "deals_imported": len(self.imported_deals),
            "last_synced_at": self.last_imported_at or datetime.datetime.utcnow().isoformat(),
            "summary": self.import_summary
        }
