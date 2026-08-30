import unittest
import os
import json
from data_sources.normalizer import normalize_deal
from data_sources.synthetic import SyntheticDataProvider
from data_sources.file_import import FileImportDataProvider
from data_sources.salesforce import SalesforceDataProvider
from data_sources.hubspot import HubSpotDataProvider
from data_sources.provider_factory import get_active_provider

class TestCRMIntegrations(unittest.TestCase):
    def test_1_canonical_normalizer_schema(self):
        raw_sf = {
            "Id": "006XX001",
            "Account": {"Name": "Acme Corp"},
            "Amount": 150000.0,
            "StageName": "Negotiation",
            "days_in_stage": 18
        }
        norm = normalize_deal(raw_sf, source_type="salesforce")
        self.assertEqual(norm["deal_id"], "006XX001")
        self.assertEqual(norm["company"], "Acme Corp")
        self.assertEqual(norm["deal_size"], 150000.0)
        self.assertEqual(norm["stage"], "Negotiation")
        self.assertIsNone(norm["response_latency_hrs"])
        self.assertIsNone(norm["sentiment_trend"])

    def test_2_file_import_csv(self):
        csv_content = b"Opportunity,Account Name,Amount,Stage,Days in Stage\nDEAL-FILE-1,Apex Global,250000,Proposal,14\n"
        provider = FileImportDataProvider()
        deals, summary = provider.process_file_content("test_pipeline.csv", csv_content)
        self.assertEqual(summary["valid_rows"], 1)
        self.assertEqual(deals[0]["deal_id"], "DEAL-FILE-1")
        self.assertEqual(deals[0]["company"], "Apex Global")
        self.assertEqual(deals[0]["deal_size"], 250000.0)

    def test_3_synthetic_provider(self):
        provider = SyntheticDataProvider()
        deals = provider.get_deals()
        self.assertGreater(len(deals), 0)
        self.assertIn("deal_id", deals[0])
        status = provider.health_check()
        self.assertTrue(status["connected"])

    def test_4_salesforce_provider_safe_unauthenticated(self):
        provider = SalesforceDataProvider()
        deals = provider.get_deals()
        self.assertEqual(len(deals), 0)
        status = provider.health_check()
        self.assertFalse(status["connected"])

    def test_5_hubspot_provider_safe_unauthenticated(self):
        provider = HubSpotDataProvider()
        deals = provider.get_deals()
        self.assertEqual(len(deals), 0)
        status = provider.health_check()
        self.assertFalse(status["connected"])

if __name__ == "__main__":
    unittest.main()
