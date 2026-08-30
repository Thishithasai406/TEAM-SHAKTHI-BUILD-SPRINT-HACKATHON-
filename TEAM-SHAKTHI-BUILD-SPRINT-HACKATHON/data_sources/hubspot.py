import os
import datetime
import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any, Optional
from data_sources.base import DealDataProvider
from data_sources.normalizer import normalize_deal

class HubSpotDataProvider(DealDataProvider):
    def __init__(self):
        self.access_token = os.getenv("HUBSPOT_ACCESS_TOKEN")
        self.base_url = "https://api.hubapi.com/crm/v3/objects/deals"
        self.last_synced_at: Optional[str] = None

    def get_deals(self) -> List[Dict[str, Any]]:
        if not self.access_token:
            return []

        url = f"{self.base_url}?limit=100&properties=dealname,amount,dealstage,createdate,hs_lastmodifieddate,closedate,pipeline&associations=contacts,companies"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        })

        deals = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                now_str = datetime.datetime.utcnow().isoformat()
                self.last_synced_at = now_str

                for r in results:
                    props = r.get("properties", {})
                    deal_id = str(r.get("id"))
                    
                    # Stakeholder count calculated from associated contacts
                    associations = r.get("associations", {})
                    contacts_assoc = associations.get("contacts", {}).get("results", [])
                    contact_count = len(contacts_assoc) if contacts_assoc else None

                    company_assoc = associations.get("companies", {}).get("results", [])
                    company_name = props.get("dealname") or f"HubSpot Deal #{deal_id}"

                    raw_hb = {
                        "deal_id": deal_id,
                        "company": company_name,
                        "deal_size": props.get("amount"),
                        "stage": props.get("dealstage", "Proposal"),
                        "close_date": props.get("closedate"),
                        "created_at": props.get("createdate"),
                        "updated_at": props.get("hs_lastmodifieddate"),
                        "source": "hubspot",
                        "source_record_id": deal_id,
                        "source_url": f"https://app.hubspot.com/contacts/deal/{deal_id}",
                        "stakeholder_count": contact_count,
                        # Unprovided CRM signals default to None
                        "response_latency_hrs": None,
                        "sentiment_trend": None,
                        "competitor_mentions": None,
                        "scope_change_flags": None
                    }
                    deals.append(normalize_deal(raw_hb, source_type="hubspot"))
        except Exception:
            pass

        return deals

    def get_deal(self, deal_id: str) -> Optional[Dict[str, Any]]:
        deals = self.get_deals()
        for d in deals:
            if d.get("deal_id") == deal_id or d.get("source_record_id") == deal_id:
                return d
        return None

    def health_check(self) -> Dict[str, Any]:
        is_conn = bool(self.access_token)
        deals_cnt = 0
        if is_conn:
            deals = self.get_deals()
            deals_cnt = len(deals)
        return {
            "source": "hubspot",
            "connected": is_conn,
            "status": "Connected" if is_conn else "Not Connected (Configure Token)",
            "deals_imported": deals_cnt,
            "last_synced_at": self.last_synced_at or datetime.datetime.utcnow().isoformat()
        }
