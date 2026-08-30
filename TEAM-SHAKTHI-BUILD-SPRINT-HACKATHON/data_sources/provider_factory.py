import os
from typing import Dict, Any
from data_sources.base import DealDataProvider
from data_sources.synthetic import SyntheticDataProvider
from data_sources.salesforce import SalesforceDataProvider
from data_sources.hubspot import HubSpotDataProvider
from data_sources.file_import import FileImportDataProvider

# Active Providers Registry
PROVIDERS: Dict[str, DealDataProvider] = {
    "synthetic": SyntheticDataProvider(),
    "salesforce": SalesforceDataProvider(),
    "hubspot": HubSpotDataProvider(),
    "file": FileImportDataProvider()
}

def get_active_data_source_type() -> str:
    """Returns configured DATA_SOURCE environment variable (default: synthetic)."""
    return os.getenv("DATA_SOURCE", "synthetic").lower().strip()

def get_active_provider() -> DealDataProvider:
    """Returns the provider instance corresponding to DATA_SOURCE configuration."""
    source_type = get_active_data_source_type()
    return PROVIDERS.get(source_type, PROVIDERS["synthetic"])
