import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from data_sources.base import DealDataProvider
from data_sources.normalizer import normalize_deal

class SyntheticDataProvider(DealDataProvider):
    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or (Path(__file__).parent.parent / "data" / "live_deals.json")
        self.deals_cache: List[Dict[str, Any]] = []
        self.load_deals()

    def load_deals(self):
        if self.data_file.exists():
            with open(self.data_file, "r") as f:
                raw_list = json.load(f)
            self.deals_cache = [normalize_deal(d, source_type="synthetic") for d in raw_list]
        else:
            self.deals_cache = []

    def get_deals(self) -> List[Dict[str, Any]]:
        if not self.deals_cache:
            self.load_deals()
        return self.deals_cache

    def get_deal(self, deal_id: str) -> Optional[Dict[str, Any]]:
        for d in self.get_deals():
            if d.get("deal_id") == deal_id:
                return d
        return None

    def health_check(self) -> Dict[str, Any]:
        return {
            "source": "synthetic",
            "connected": True,
            "status": "Connected (Demo Mode)",
            "deals_imported": len(self.get_deals()),
            "last_synced_at": datetime.datetime.utcnow().isoformat()
        }
