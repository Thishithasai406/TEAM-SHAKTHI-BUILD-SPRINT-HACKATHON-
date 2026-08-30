import os
import datetime
import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any, Optional
from data_sources.base import DealDataProvider
from data_sources.normalizer import normalize_deal

class SalesforceDataProvider(DealDataProvider):
    def __init__(self):
        self.client_id = os.getenv("SALESFORCE_CLIENT_ID")
        self.client_secret = os.getenv("SALESFORCE_CLIENT_SECRET")
        self.username = os.getenv("SALESFORCE_USERNAME")
        self.password = os.getenv("SALESFORCE_PASSWORD")
        self.security_token = os.getenv("SALESFORCE_SECURITY_TOKEN", "")
        self.login_url = os.getenv("SALESFORCE_LOGIN_URL", "https://login.salesforce.com").rstrip("/")
        
        self.access_token: Optional[str] = None
        self.instance_url: Optional[str] = None
        self.last_synced_at: Optional[str] = None

    def authenticate(self) -> bool:
        if not (self.client_id and self.client_secret and self.username and self.password):
            return False
            
        token_url = f"{self.login_url}/services/oauth2/token"
        payload = urllib.parse.urlencode({
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": f"{self.password}{self.security_token}"
        }).encode("utf-8")

        req = urllib.request.Request(token_url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                self.access_token = res_data.get("access_token")
                self.instance_url = res_data.get("instance_url")
                return True
        except Exception:
            return False

    def get_deals(self) -> List[Dict[str, Any]]:
        if not self.access_token:
            if not self.authenticate():
                return []

        soql = "SELECT Id, Name, Amount, StageName, CloseDate, Probability, CreatedDate, LastModifiedDate, Account.Name FROM Opportunity LIMIT 200"
        query_url = f"{self.instance_url}/services/data/v57.0/query?q={urllib.parse.quote(soql)}"
        
        req = urllib.request.Request(query_url, headers={"Authorization": f"Bearer {self.access_token}"})
        deals = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                records = data.get("records", [])
                now_str = datetime.datetime.utcnow().isoformat()
                self.last_synced_at = now_str
                
                for r in records:
                    account_name = "Salesforce Account"
                    if r.get("Account") and isinstance(r["Account"], dict):
                        account_name = r["Account"].get("Name", "Salesforce Account")

                    raw_sf = {
                        "deal_id": r.get("Id"),
                        "company": account_name,
                        "deal_size": r.get("Amount"),
                        "stage": r.get("StageName"),
                        "close_date": r.get("CloseDate"),
                        "probability": r.get("Probability"),
                        "created_at": r.get("CreatedDate"),
                        "updated_at": r.get("LastModifiedDate"),
                        "source": "salesforce",
                        "source_record_id": r.get("Id"),
                        "source_url": f"{self.instance_url}/{r.get('Id')}" if self.instance_url else None,
                        # Unprovided CRM signals default to None
                        "response_latency_hrs": None,
                        "sentiment_trend": None,
                        "stakeholder_count": None,
                        "competitor_mentions": None,
                        "scope_change_flags": None
                    }
                    deals.append(normalize_deal(raw_sf, source_type="salesforce"))
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
        is_conn = self.authenticate() if not self.access_token else True
        return {
            "source": "salesforce",
            "connected": is_conn,
            "status": "Connected" if is_conn else "Not Connected (Configure Credentials)",
            "deals_imported": len(self.get_deals()) if is_conn else 0,
            "last_synced_at": self.last_synced_at or datetime.datetime.utcnow().isoformat()
        }
