from typing import Dict, Any, List, Optional

class DealDataProvider:
    def get_deals(self) -> List[Dict[str, Any]]:
        """Fetch all active live deals in canonical format."""
        raise NotImplementedError

    def get_deal(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single deal by deal_id in canonical format."""
        raise NotImplementedError

    def health_check(self) -> Dict[str, Any]:
        """Check provider connectivity and status."""
        raise NotImplementedError
