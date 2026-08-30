import unittest
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient
from main import app
from feature_extractor import extract_engagement_features, analyze_text_nlp
from deal_scorer import scorer
from recommendation_engine import generate_recovery_recommendation

class TestDealIQExtendedChecks(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_1_data_leakage_prevention(self):
        """Confirm the Random Forest model was trained ONLY on historical set and NEVER saw live set outcomes."""
        hist_path = Path(__file__).parent / "data" / "historical_deals.json"
        live_path = Path(__file__).parent / "data" / "live_deals.json"
        
        with open(hist_path, "r") as f:
            hist_deals = json.load(f)
        with open(live_path, "r") as f:
            live_deals = json.load(f)

        # Confirm scorer training set size equals historical set size
        self.assertEqual(len(scorer.historical_deals), len(hist_deals))
        self.assertEqual(len(scorer.historical_vectors), 550)

        # Confirm all live deals have hidden outcomes ('in_progress')
        for deal in live_deals:
            self.assertEqual(deal.get("outcome"), "in_progress")

        # Confirm no live deal ID exists in the historical training set
        hist_ids = {d["deal_id"] for d in hist_deals}
        live_ids = {d["deal_id"] for d in live_deals}
        intersection = hist_ids.intersection(live_ids)
        self.assertEqual(len(intersection), 0, f"Data leakage detected! Shared IDs: {intersection}")

    def test_2_llm_key_unset_fallback(self):
        """Confirm feature_extractor falls back gracefully to NLP heuristics when OPENAI_API_KEY is unset."""
        original_key = os.environ.get("OPENAI_API_KEY")
        try:
            # Unset API Key
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            raw_deal = {
                "deal_id": "FALLBACK-01",
                "transcript": "Call with Salesforce competitor rep and CTO John. Scope change requested."
            }

            extracted = extract_engagement_features(raw_deal, use_llm=True)
            self.assertEqual(extracted.extractor_used, "heuristic")
            self.assertIn("Salesforce", extracted.competitors_found)
            self.assertGreater(len(extracted.scope_change_phrases), 0)
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_3_zero_winning_analogs_recommendation_robustness(self):
        """Confirm recommendation_engine does not crash when nearest analogs contain ZERO 'won' deals."""
        all_lost_analogs = [
            {"deal_id": "DEAL-0001", "outcome": "lost", "deal_size": 100000},
            {"deal_id": "DEAL-0002", "outcome": "stalled", "deal_size": 200000},
            {"deal_id": "DEAL-0003", "outcome": "lost", "deal_size": 150000},
            {"deal_id": "DEAL-0004", "outcome": "stalled", "deal_size": 300000},
            {"deal_id": "DEAL-0005", "outcome": "lost", "deal_size": 250000}
        ]

        raw_deal = {
            "deal_id": "HIGH-RISK-DEAL",
            "deal_size": 500000,
            "competitor_mentions": 3,
            "stakeholder_count": 1,
            "days_in_stage": 95
        }

        rec = generate_recovery_recommendation(raw_deal, risk_score=90.0, nearest_analogs=all_lost_analogs)
        
        self.assertFalse(rec["has_recovered_analogs"])
        self.assertEqual(rec["recovered_analogs_count"], 0)
        self.assertIsNotNone(rec["reason"])
        self.assertIsNotNone(rec["recommendation"])
        self.assertIsNotNone(rec["action_type"])
        self.assertTrue(len(rec["recommendation"]) > 0)

    def test_4_clean_installation_and_app_execution(self):
        """Confirm app runs end-to-end with status code 200 across main endpoints and calculates dollar-value-at-risk."""
        metrics_res = self.client.get("/api/metrics")
        self.assertEqual(metrics_res.status_code, 200)
        metrics_data = metrics_res.json()
        self.assertIn("value_at_risk", metrics_data)
        self.assertGreater(metrics_data["value_at_risk"], 0.0)

        live_res = self.client.get("/api/deals/live")
        self.assertEqual(live_res.status_code, 200)
        live_data = live_res.json()
        self.assertIn("last_refreshed", live_data)
        self.assertIn("deals", live_data)
        self.assertEqual(len(live_data["deals"]), 100)

    def test_5_stage_label_consistency_across_all_six_stages(self):
        """Verify all 6 pipeline stages map accurately without fallback to Evaluation."""
        stages = ["Discovery", "Qualification", "Evaluation", "Proposal", "Negotiation", "Closing"]
        for stg in stages:
            raw_deal = {
                "deal_id": f"TEST-{stg}",
                "company_name": f"{stg} Test Corp",
                "stage": stg,
                "deal_size": 300000,
                "stakeholder_count": 1,
                "competitor_mentions": 3,
                "days_in_stage": 40
            }
            rec = generate_recovery_recommendation(raw_deal, risk_score=85.0, nearest_analogs=[])
            self.assertEqual(rec["stage_category"], stg)
            self.assertTrue(rec["action_type"].startswith(stg), f"Expected action_type to start with '{stg}', got '{rec['action_type']}'")

    def test_6_pipeline_at_risk_metrics_and_deduplication(self):
        """Verify pipeline at risk metrics sum correctly and deduplicate live deals."""
        metrics_res = self.client.get("/api/metrics")
        self.assertEqual(metrics_res.status_code, 200)
        data = metrics_res.json()
        
        self.assertIn("total_pipeline_value", data)
        self.assertIn("value_at_risk", data)
        self.assertIn("high_risk_value", data)
        self.assertIn("medium_risk_value", data)
        self.assertIn("low_risk_value", data)

        # Confirm value_at_risk = high_risk_value + medium_risk_value
        self.assertEqual(data["value_at_risk"], round(data["high_risk_value"] + data["medium_risk_value"], 2))
        # Confirm total_pipeline_value = high + medium + low
        self.assertEqual(data["total_pipeline_value"], round(data["high_risk_value"] + data["medium_risk_value"] + data["low_risk_value"], 2))

        hist_res = self.client.get("/api/deals/historical")
        self.assertEqual(hist_res.status_code, 200)
        self.assertEqual(len(hist_res.json()), 550)

        index_res = self.client.get("/")
        self.assertEqual(index_res.status_code, 200)

    def test_7_outranking_and_root_cause_scenarios(self):
        """Test root-cause ranking, outranking explanations, and positive signal weighting."""
        
        # Scenario 1: High competitor mentions + declining sentiment -> Competitive Pressure
        d1 = {"deal_id": "D1", "competitor_mentions": 4, "sentiment_trend": "declining", "stakeholder_count": 3, "days_in_stage": 15, "response_latency_hrs": 12.0}
        rec1 = generate_recovery_recommendation(d1, risk_score=80.0, nearest_analogs=[])
        self.assertEqual(rec1["root_cause"], "Competitive Pressure")
        why_text1 = " ".join(rec1["why_this_move"]) if isinstance(rec1["why_this_move"], list) else rec1["why_this_move"]
        self.assertIn("Competitive Pressure is the primary root cause", why_text1)

        # Scenario 2: High competitor mentions + healthy sentiment + strong stakeholder coverage -> deprioritize Competitive Pressure if single-threading is present
        d2 = {"deal_id": "D2", "competitor_mentions": 1, "sentiment_trend": "improving", "stakeholder_count": 1, "deal_size": 300000, "days_in_stage": 20, "response_latency_hrs": 10.0}
        rec2 = generate_recovery_recommendation(d2, risk_score=75.0, nearest_analogs=[])
        self.assertEqual(rec2["root_cause"], "Lack of Stakeholder Alignment")

        # Scenario 3: Multiple scope changes + scope-driven delay -> Scope Instability
        d3 = {"deal_id": "D3", "scope_change_flags": 3, "days_in_stage": 55, "stakeholder_count": 4, "response_latency_hrs": 15.0, "competitor_mentions": 0}
        rec3 = generate_recovery_recommendation(d3, risk_score=70.0, nearest_analogs=[])
        self.assertEqual(rec3["root_cause"], "Scope Instability")

        # Scenario 4: Single stakeholder + otherwise healthy communication -> Lack of Stakeholder Alignment
        d4 = {"deal_id": "D4", "stakeholder_count": 1, "deal_size": 250000, "response_latency_hrs": 12.0, "sentiment_trend": "improving", "competitor_mentions": 0}
        rec4 = generate_recovery_recommendation(d4, risk_score=65.0, nearest_analogs=[])
        self.assertEqual(rec4["root_cause"], "Lack of Stakeholder Alignment")

        # Scenario 5: High response latency + declining sentiment -> Buyer Disengagement
        d5 = {"deal_id": "D5", "response_latency_hrs": 55.0, "sentiment_trend": "declining", "stakeholder_count": 4, "competitor_mentions": 0}
        rec5 = generate_recovery_recommendation(d5, risk_score=85.0, nearest_analogs=[])
        self.assertEqual(rec5["root_cause"], "Buyer Disengagement")

        # Scenario 6: Healthy response latency + improving sentiment -> Buyer Disengagement deprioritized
        d6 = {"deal_id": "D6", "response_latency_hrs": 12.0, "sentiment_trend": "improving", "stakeholder_count": 4, "days_in_stage": 75}
        rec6 = generate_recovery_recommendation(d6, risk_score=50.0, nearest_analogs=[])
        self.assertNotEqual(rec6["root_cause"], "Buyer Disengagement")

        # Scenario 7: Procurement evidence -> Procurement Delay
        d7 = {"deal_id": "D7", "stage": "Negotiation", "days_in_stage": 35, "response_latency_hrs": 35.0, "stakeholder_count": 4, "competitor_mentions": 0}
        rec7 = generate_recovery_recommendation(d7, risk_score=75.0, nearest_analogs=[])
        self.assertEqual(rec7["root_cause"], "Procurement Delay")

        # Scenario 8: Long stage duration alone without root cause signals -> never use Stage Stagnation when underlying cause exists
        d8 = {"deal_id": "D8", "stage": "Proposal", "days_in_stage": 80, "stakeholder_count": 1, "deal_size": 300000, "response_latency_hrs": 15.0}
        rec8 = generate_recovery_recommendation(d8, risk_score=80.0, nearest_analogs=[])
        self.assertEqual(rec8["root_cause"], "Lack of Stakeholder Alignment")

        # Scenario 9 & 10: Multiple competing causes -> explicit outranking explanation and positive signals acknowledged in WHY THIS MOVE
        d9 = {"deal_id": "D9", "stage": "Proposal", "days_in_stage": 82, "response_latency_hrs": 4.7, "sentiment_trend": "declining", "competitor_mentions": 5, "scope_change_flags": 4, "stakeholder_count": 4}
        rec9 = generate_recovery_recommendation(d9, risk_score=95.0, nearest_analogs=[])
        self.assertEqual(rec9["root_cause"], "Competitive Pressure")
        why_text9 = " ".join(rec9["why_this_move"]) if isinstance(rec9["why_this_move"], list) else rec9["why_this_move"]
        self.assertIn("Competing factor:", why_text9)
        self.assertIn("Positive signals considered:", why_text9)

    def test_8_why_this_move_formatting_across_all_causes(self):
        """Verify why_this_move returns a clean structured array across all root causes matching all copy-quality rules."""
        causes_deals = [
            ("Competitive Pressure", {"competitor_mentions": 4, "sentiment_trend": "declining"}),
            ("Scope Instability", {"scope_change_flags": 3, "days_in_stage": 50}),
            ("Lack of Stakeholder Alignment", {"stakeholder_count": 1, "deal_size": 300000}),
            ("Buyer Disengagement", {"response_latency_hrs": 60, "sentiment_trend": "declining"}),
            ("Procurement Delay", {"stage": "Negotiation", "days_in_stage": 35}),
            ("Deal Stagnation", {"stage": "Proposal", "days_in_stage": 90})
        ]

        for cause_name, deal_props in causes_deals:
            d = {"deal_id": f"TEST-{cause_name}", "company_name": "Test Co", "stage": "Proposal", **deal_props}
            rec = generate_recovery_recommendation(d, risk_score=80.0, nearest_analogs=[])
            
            self.assertIsInstance(rec["why_this_move"], list)
            self.assertEqual(len(rec["why_this_move"]), 4, f"Expected exactly 4 items in why_this_move list for {cause_name}")
            
            # Verify clean copy rules for each item
            for bullet in rec["why_this_move"]:
                self.assertIsInstance(bullet, str)
                self.assertGreater(len(bullet.strip()), 0, "Bullet item should not be empty")
                self.assertFalse(bullet.startswith("-"), f"Found raw leading dash in: {bullet}")
                self.assertFalse(bullet.startswith("•"), f"Found raw leading bullet in: {bullet}")
                self.assertFalse(bullet.startswith("*"), f"Found raw leading asterisk in: {bullet}")
                self.assertNotIn("**", bullet, f"Found raw markdown bold in: {bullet}")
                self.assertNotIn("..", bullet, f"Found double periods in: {bullet}")
                self.assertNotIn(".,", bullet, f"Found period-comma typo in: {bullet}")
                self.assertTrue(bullet.endswith("."), f"Bullet should end with a period: {bullet}")
                self.assertFalse(bullet.endswith(".."), f"Bullet should not end with double periods: {bullet}")

            # Verify why_not_wait is separate and not in why_this_move
            self.assertNotIn(rec["why_not_wait"], rec["why_this_move"])

    def test_9_comprehensive_twenty_scenario_validation(self):
        """Test all 20 required root-cause and validation scenarios."""
        # 1. Single-threaded large deal
        rec1 = generate_recovery_recommendation({"deal_id": "T1", "stakeholder_count": 1, "deal_size": 400000}, risk_score=80.0, nearest_analogs=[])
        self.assertEqual(rec1["root_cause"], "Lack of Stakeholder Alignment")

        # 2. Strong multi-threading + slow response
        rec2 = generate_recovery_recommendation({"deal_id": "T2", "stakeholder_count": 5, "response_latency_hrs": 55.0, "sentiment_trend": "declining"}, risk_score=75.0, nearest_analogs=[])
        self.assertEqual(rec2["root_cause"], "Buyer Disengagement")

        # 3. Strong multi-threading + fast response
        rec3 = generate_recovery_recommendation({"deal_id": "T3", "stakeholder_count": 5, "response_latency_hrs": 10.0, "sentiment_trend": "improving"}, risk_score=30.0, nearest_analogs=[])
        self.assertEqual(rec3["root_cause"], "Early Stage Qualification Friction")

        # 4. Competitors + declining sentiment
        rec4 = generate_recovery_recommendation({"deal_id": "T4", "competitor_mentions": 3, "sentiment_trend": "declining", "stakeholder_count": 4}, risk_score=85.0, nearest_analogs=[])
        self.assertEqual(rec4["root_cause"], "Competitive Pressure")

        # 5. Competitors + improving sentiment + strong coverage
        rec5 = generate_recovery_recommendation({"deal_id": "T5", "competitor_mentions": 1, "sentiment_trend": "improving", "stakeholder_count": 4, "response_latency_hrs": 12.0}, risk_score=40.0, nearest_analogs=[])
        self.assertNotEqual(rec5["root_cause"], "Competitive Pressure")

        # 6. Zero competitors
        rec6 = generate_recovery_recommendation({"deal_id": "T6", "competitor_mentions": 0, "stakeholder_count": 1}, risk_score=60.0, nearest_analogs=[])
        self.assertNotEqual(rec6["root_cause"], "Competitive Pressure")

        # 7. One scope change alone
        rec7 = generate_recovery_recommendation({"deal_id": "T7", "scope_change_flags": 1, "days_in_stage": 15, "stakeholder_count": 4, "response_latency_hrs": 12.0}, risk_score=35.0, nearest_analogs=[])
        self.assertNotEqual(rec7["root_cause"], "Scope Instability")

        # 8. Multiple scope changes
        rec8 = generate_recovery_recommendation({"deal_id": "T8", "scope_change_flags": 3, "days_in_stage": 50, "stakeholder_count": 4, "response_latency_hrs": 12.0}, risk_score=75.0, nearest_analogs=[])
        self.assertEqual(rec8["root_cause"], "Scope Instability")

        # 9. Scope changes + competitive pressure
        rec9 = generate_recovery_recommendation({"deal_id": "T9", "scope_change_flags": 3, "competitor_mentions": 4, "sentiment_trend": "declining", "stakeholder_count": 4}, risk_score=90.0, nearest_analogs=[])
        self.assertIn(rec9["root_cause"], ["Competitive Pressure", "Scope Instability"])
        self.assertIsNotNone(rec9["competing_root_cause"])

        # 10. High latency + declining sentiment
        rec10 = generate_recovery_recommendation({"deal_id": "T10", "response_latency_hrs": 60.0, "sentiment_trend": "declining", "stakeholder_count": 4}, risk_score=80.0, nearest_analogs=[])
        self.assertEqual(rec10["root_cause"], "Buyer Disengagement")

        # 11. Fast latency + improving sentiment
        rec11 = generate_recovery_recommendation({"deal_id": "T11", "response_latency_hrs": 8.0, "sentiment_trend": "improving", "stakeholder_count": 4}, risk_score=25.0, nearest_analogs=[])
        self.assertNotEqual(rec11["root_cause"], "Buyer Disengagement")

        # 12. Closing + stage stagnation only
        rec12 = generate_recovery_recommendation({"deal_id": "T12", "stage": "Closing", "days_in_stage": 100, "stakeholder_count": 4, "response_latency_hrs": 12.0}, risk_score=60.0, nearest_analogs=[])
        self.assertNotEqual(rec12["root_cause"], "Procurement Delay")

        # 13. Closing + explicit procurement friction
        rec13 = generate_recovery_recommendation({"deal_id": "T13", "stage": "Closing", "days_in_stage": 35, "response_latency_hrs": 35.0, "stakeholder_count": 4}, risk_score=75.0, nearest_analogs=[])
        self.assertEqual(rec13["root_cause"], "Procurement Delay")

        # 14. Negotiation + procurement friction
        rec14 = generate_recovery_recommendation({"deal_id": "T14", "stage": "Negotiation", "days_in_stage": 30, "response_latency_hrs": 32.0, "stakeholder_count": 4}, risk_score=75.0, nearest_analogs=[])
        self.assertEqual(rec14["root_cause"], "Procurement Delay")

        # 15. Negotiation + no procurement evidence
        rec15 = generate_recovery_recommendation({"deal_id": "T15", "stage": "Negotiation", "days_in_stage": 10, "response_latency_hrs": 12.0, "stakeholder_count": 4}, risk_score=30.0, nearest_analogs=[])
        self.assertNotEqual(rec15["root_cause"], "Procurement Delay")

        # 16. Early stage qualification friction
        rec16 = generate_recovery_recommendation({"deal_id": "T16", "stage": "Discovery", "days_in_stage": 10, "stakeholder_count": 4, "response_latency_hrs": 12.0}, risk_score=20.0, nearest_analogs=[])
        self.assertEqual(rec16["root_cause"], "Early Stage Qualification Friction")

        # 17. Discovery with no qualification evidence
        rec17 = generate_recovery_recommendation({"deal_id": "T17", "stage": "Discovery", "days_in_stage": 5, "stakeholder_count": 3, "response_latency_hrs": 10.0}, risk_score=15.0, nearest_analogs=[])
        self.assertEqual(rec17["root_cause"], "Early Stage Qualification Friction")

        # 18. Multiple competing root causes
        rec18 = generate_recovery_recommendation({"deal_id": "T18", "stage": "Proposal", "days_in_stage": 80, "stakeholder_count": 1, "competitor_mentions": 4, "sentiment_trend": "declining"}, risk_score=95.0, nearest_analogs=[])
        self.assertTrue(len(rec18["competing_root_cause"]) > 0)
        self.assertIn("Competing factor:", " ".join(rec18["why_this_move"]))

        # 19. Positive signals contradicting an apparent root cause
        rec19 = generate_recovery_recommendation({"deal_id": "T19", "stage": "Proposal", "competitor_mentions": 1, "sentiment_trend": "improving", "response_latency_hrs": 8.0, "stakeholder_count": 5}, risk_score=40.0, nearest_analogs=[])
        self.assertIn("Positive signals considered:", " ".join(rec19["why_this_move"]))

        # 20. Confirm stakeholder terminology is always Stakeholders Involved: N
        rec20 = generate_recovery_recommendation({"deal_id": "T20", "stakeholder_count": 1}, risk_score=70.0, nearest_analogs=[])
        for b in rec20["why_this_move"]:
            self.assertNotIn("champions", b.lower())
            self.assertNotIn("economic buyer", b.lower())

    def test_10_evidence_grounding_and_no_hallucinated_claims(self):
        """Verify that generated text never hallucinated unobserved evidence (redlines, technical friction, roles)."""
        # Test deal without explicit redline field
        d_proc = {"deal_id": "PROC-TEST", "stage": "Negotiation", "days_in_stage": 35, "response_latency_hrs": 55.0, "stakeholder_count": 7, "competitor_mentions": 4, "scope_change_flags": 1, "sentiment_trend": "stable"}
        rec_proc = generate_recovery_recommendation(d_proc, risk_score=80.0, nearest_analogs=[])
        
        self.assertEqual(rec_proc["root_cause"], "Procurement Delay")
        
        # Verify no hallucinated redline/legal claims when not present in source data
        why_text = " ".join(rec_proc["why_this_move"])
        self.assertNotIn("redlines are blocking", why_text.lower())
        self.assertIn("response latency", why_text.lower())
        self.assertIn("evidence_trace", rec_proc)

        # Verify no inferred stakeholder roles across generated text
        for text in [rec_proc["primary_action"], rec_proc["secondary_action"]] + rec_proc["why_this_move"]:
            self.assertNotIn("champions", text.lower())
            self.assertNotIn("economic buyer", text.lower())

    def test_11_health_score_comprehensive_suite(self):
        """Automated test suite verifying Health Score requirements 1 through 16."""
        from main import calculate_deal_health, load_json_data

        # 1. Healthy deal -> high Health Score
        healthy_deal = {
            "stage": "Proposal",
            "days_in_stage": 10,
            "response_latency_hrs": 8.0,
            "sentiment_trend": "improving",
            "stakeholder_count": 5,
            "competitor_mentions": 0,
            "scope_change_flags": 0
        }
        h_healthy = calculate_deal_health(healthy_deal)
        self.assertGreaterEqual(h_healthy["health_score"], 80.0)

        # 2. Severely unhealthy deal -> low Health Score
        unhealthy_deal = {
            "stage": "Proposal",
            "days_in_stage": 120,
            "response_latency_hrs": 80.0,
            "sentiment_trend": "declining",
            "stakeholder_count": 1,
            "competitor_mentions": 4,
            "scope_change_flags": 3
        }
        h_unhealthy = calculate_deal_health(unhealthy_deal)
        self.assertLessEqual(h_unhealthy["health_score"], 20.0)

        # 3. Improving sentiment increases health
        d_base = {"stage": "Proposal", "days_in_stage": 15, "response_latency_hrs": 20.0, "sentiment_trend": "stable"}
        d_imp = {**d_base, "sentiment_trend": "improving"}
        self.assertGreater(calculate_deal_health(d_imp)["health_score"], calculate_deal_health(d_base)["health_score"])

        # 4. Declining sentiment decreases health
        d_dec = {**d_base, "sentiment_trend": "declining"}
        self.assertLess(calculate_deal_health(d_dec)["health_score"], calculate_deal_health(d_base)["health_score"])

        # 5. Healthy response latency improves health
        d_fast_lat = {**d_base, "response_latency_hrs": 5.0}
        d_slow_lat = {**d_base, "response_latency_hrs": 30.0}
        self.assertGreater(calculate_deal_health(d_fast_lat)["health_score"], calculate_deal_health(d_slow_lat)["health_score"])

        # 6. High response latency decreases health
        d_high_lat = {**d_base, "response_latency_hrs": 75.0}
        self.assertLess(calculate_deal_health(d_high_lat)["health_score"], calculate_deal_health(d_base)["health_score"])

        # 7. Excessive stage duration decreases health
        d_normal_stage = {**d_base, "days_in_stage": 15}
        d_stale_stage = {**d_base, "days_in_stage": 100}
        self.assertLess(calculate_deal_health(d_stale_stage)["health_score"], calculate_deal_health(d_normal_stage)["health_score"])

        # 8. Zero competitors is better than significant competitor presence
        d_zero_comp = {**d_base, "competitor_mentions": 0}
        d_high_comp = {**d_base, "competitor_mentions": 4}
        self.assertGreater(calculate_deal_health(d_zero_comp)["health_score"], calculate_deal_health(d_high_comp)["health_score"])

        # 9. Multiple scope changes decrease health
        d_zero_scope = {**d_base, "scope_change_flags": 0}
        d_multi_scope = {**d_base, "scope_change_flags": 3}
        self.assertLess(calculate_deal_health(d_multi_scope)["health_score"], calculate_deal_health(d_zero_scope)["health_score"])

        # 10. Strong stakeholder coverage improves health
        d_high_sh = {**d_base, "stakeholder_count": 6}
        d_mid_sh = {**d_base, "stakeholder_count": 2}
        self.assertGreater(calculate_deal_health(d_high_sh)["health_score"], calculate_deal_health(d_mid_sh)["health_score"])

        # 11. Insufficient stakeholder coverage decreases health
        d_low_sh = {**d_base, "stakeholder_count": 1}
        self.assertLess(calculate_deal_health(d_low_sh)["health_score"], calculate_deal_health(d_high_sh)["health_score"])

        # 12. Score never goes below 0
        extreme_unhealthy = {
            "days_in_stage": 500,
            "response_latency_hrs": 200.0,
            "sentiment_trend": "declining",
            "stakeholder_count": 0,
            "competitor_mentions": 10,
            "scope_change_flags": 10
        }
        self.assertGreaterEqual(calculate_deal_health(extreme_unhealthy)["health_score"], 0.0)

        # 13. Score never exceeds 100
        extreme_healthy = {
            "days_in_stage": 1,
            "response_latency_hrs": 1.0,
            "sentiment_trend": "improving",
            "stakeholder_count": 10,
            "competitor_mentions": 0,
            "scope_change_flags": 0
        }
        self.assertLessEqual(calculate_deal_health(extreme_healthy)["health_score"], 100.0)

        # 14. Same input always produces same score (Determinism)
        res1 = calculate_deal_health(healthy_deal)
        res2 = calculate_deal_health(healthy_deal)
        self.assertEqual(res1["health_score"], res2["health_score"])
        self.assertEqual(res1["health_factors"], res2["health_factors"])

        # 15. Backend score exactly matches frontend displayed score (via API endpoint)
        live_res = self.client.get("/api/deals/live")
        self.assertEqual(live_res.status_code, 200)
        live_deals = live_res.json()["deals"]
        for live_d in live_deals[:10]:
            backend_calc = calculate_deal_health(live_d)
            self.assertEqual(live_d["health_score"], backend_calc["health_score"])

        # 16. Historical deals can be recalculated consistently
        hist_deals = load_json_data("historical_deals.json")
        for hist_d in hist_deals[:20]:
            calc1 = calculate_deal_health(hist_d)
            calc2 = calculate_deal_health(hist_d)
            self.assertEqual(calc1["health_score"], calc2["health_score"])

    def test_12_ml_risk_model_regression_and_validation(self):
        """Automated regression & validation tests for ML Risk Scorer in deal_scorer.py."""
        from deal_scorer import scorer, extract_normalized_features
        from main import load_json_data

        # 1. Correct class mapping check
        classes = list(scorer.clf.classes_)
        self.assertIn(0, classes)
        self.assertIn(1, classes)

        # 2. Risk percentage is bounded between 0 and 100
        live_deals = load_json_data("live_deals.json")
        for d in live_deals:
            s_res = scorer.score_deal(d)
            risk = s_res["risk_score"]
            self.assertGreaterEqual(risk, 0.0)
            self.assertLessEqual(risk, 100.0)
            self.assertIn(s_res["risk_category"], ["Low Risk", "Medium Risk", "High Risk"])

        # 3. Model does NOT return 100% for every deal
        scores = [scorer.score_deal(d)["risk_score"] for d in live_deals]
        self.assertTrue(any(s < 100.0 for s in scores), "Model incorrectly saturated at 100% for all deals")
        self.assertTrue(any(s < 90.0 for s in scores), "Model should produce risk scores below 90%")

        # 4. Different feature vectors produce meaningfully different risk scores
        low_risk_deal = {
            "deal_size": 50000.0,
            "days_in_stage": 5,
            "response_latency_hrs": 2.0,
            "stakeholder_count": 10,
            "competitor_mentions": 0,
            "scope_change_flags": 0,
            "sentiment_trend": "improving"
        }
        high_risk_deal = {
            "deal_size": 1200000.0,
            "days_in_stage": 150,
            "response_latency_hrs": 70.0,
            "stakeholder_count": 1,
            "competitor_mentions": 5,
            "scope_change_flags": 4,
            "sentiment_trend": "declining"
        }
        r_low = scorer.score_deal(low_risk_deal)["risk_score"]
        r_high = scorer.score_deal(high_risk_deal)["risk_score"]
        self.assertGreater(r_high, r_low, f"High risk deal ({r_high}%) should have higher risk than low risk deal ({r_low}%)")

        # 5. k-NN historical analogs correctness
        # Test that self (live deal) cannot appear as its own neighbor if present in dataset
        hist_sample = load_json_data("historical_deals.json")[0]
        knn_res = scorer.score_deal(hist_sample, k=5)
        neighbor_ids = [a["deal_id"] for a in knn_res["nearest_analogs"]]
        self.assertNotIn(hist_sample["deal_id"], neighbor_ids, "Live deal itself must not appear as a historical analog neighbor")
        self.assertEqual(len(knn_res["nearest_analogs"]), 5)

        # 6. API response format compatibility
        api_res = self.client.get("/api/deals/live")
        self.assertEqual(api_res.status_code, 200)
        d_sample = api_res.json()["deals"][0]
        self.assertIn("risk_score", d_sample)
        self.assertIn("risk_category", d_sample)
        self.assertIn("nearest_analogs", d_sample)
        self.assertIn("model_type", d_sample)

    def test_13_advanced_ml_validation_metrics_and_calibration(self):
        """Automated tests for probability thresholding, distribution report, Brier skill score, and ECE."""
        from deal_scorer import scorer, calculate_ece
        from main import load_json_data

        live_deals = load_json_data("live_deals.json")
        report = scorer.get_before_after_distribution_report(live_deals)

        # 1. Distribution report structure checks
        self.assertIn("historical_deals", report)
        self.assertIn("live_deals", report)
        
        live_stats = report["live_deals"]["after_calibrated"]
        self.assertIn("min", live_stats)
        self.assertIn("max", live_stats)
        self.assertIn("mean", live_stats)
        self.assertIn("median", live_stats)
        self.assertIn("std", live_stats)
        self.assertIn("p25", live_stats)
        self.assertIn("p75", live_stats)

        # Risk level percentages sum to 100%
        sum_pct = live_stats["high_risk"]["pct"] + live_stats["medium_risk"]["pct"] + live_stats["low_risk"]["pct"]
        self.assertAlmostEqual(sum_pct, 100.0, delta=0.5)

        # 2. Validation Report metrics checks
        val_rep = scorer.validation_report
        self.assertIn("threshold_search_cv", val_rep)
        self.assertIn("holdout_evaluation", val_rep)
        
        holdout = val_rep["holdout_evaluation"]
        self.assertIn("roc_auc", holdout)
        self.assertIn("pr_auc", holdout)
        self.assertIn("baseline_pr_auc", holdout)
        self.assertIn("brier_score", holdout)
        self.assertIn("brier_skill_score", holdout)
        self.assertIn("ece", holdout)
        
        # Verify Brier skill score is positive (outperforms naive prevalence baseline)
        self.assertGreater(holdout["brier_skill_score"], 0.0)

        # 3. Decision threshold search checks
        cv_search = val_rep["threshold_search_cv"]
        self.assertIn("0.80", cv_search)
        th_80 = cv_search["0.80"]
        self.assertGreater(th_80["specificity"], 0.30)
        self.assertIn("confusion_matrix", th_80)

        # 4. Leakage & Candidate Model comparison checks
        self.assertIn("candidate_models_comparison", val_rep)
        cand_comp = val_rep["candidate_models_comparison"]
        self.assertIn("Calibrated Random Forest (max_depth=5, balanced)", cand_comp)
        self.assertIn("Calibrated Logistic Regression (balanced)", cand_comp)
        
        # Verify zero shared IDs between training set and live pipeline
        hist_deals = load_json_data("historical_deals.json")
        hist_ids = {d["deal_id"] for d in hist_deals}
        live_ids = {d["deal_id"] for d in live_deals}
        self.assertEqual(len(hist_ids.intersection(live_ids)), 0)

    def test_14_root_cause_and_risk_ranking_strict_alignment(self):
        """Automated regression tests verifying Requirements 1-8 for Root Cause, Recommendation, and Risk Ranking alignment."""
        from recommendation_engine import generate_recovery_recommendation

        # Test Scenario 1: Extended Time in Stage is highest risk signal -> Root cause MUST be Extended Time in Stage / Deal Stagnation
        d1 = {
            "deal_id": "DEAL-0596",
            "company_name": "BlueWave Tech",
            "stage": "Closing",
            "days_in_stage": 175,
            "response_latency_hrs": 20.8,
            "sentiment_trend": "declining",
            "stakeholder_count": 3,
            "competitor_mentions": 5,
            "scope_change_flags": 4,
            "deal_size": 450000
        }
        rec1 = generate_recovery_recommendation(d1, risk_score=88.0, nearest_analogs=[])
        primary_risk_signal1 = rec1["ranked_risks"]["primary"]["factor_name"]
        self.assertEqual(primary_risk_signal1, "Extended Time in Stage")
        self.assertEqual(rec1["root_cause"], "Extended Time in Stage / Deal Stagnation")
        self.assertIn("executive sponsor", rec1["primary_action"].lower())

        # Test Scenario 2: Competitive Pressure highest -> Competitive Pressure root cause & recommendation
        d2 = {
            "deal_id": "D-COMP",
            "stage": "Proposal",
            "days_in_stage": 10,
            "competitor_mentions": 4,
            "sentiment_trend": "declining",
            "stakeholder_count": 4,
            "response_latency_hrs": 12.0
        }
        rec2 = generate_recovery_recommendation(d2, risk_score=85.0, nearest_analogs=[])
        primary_risk_signal2 = rec2["ranked_risks"]["primary"]["factor_name"]
        self.assertEqual(primary_risk_signal2, "Competitor Presence")
        self.assertEqual(rec2["root_cause"], "Competitive Pressure")
        self.assertIn("value matrix", rec2["primary_action"].lower())

        # Test Scenario 3: Scope Instability highest -> Scope Instability root cause & freeze scope recommendation
        d3 = {
            "deal_id": "D-SCOPE",
            "stage": "Proposal",
            "days_in_stage": 10,
            "scope_change_flags": 4,
            "stakeholder_count": 4,
            "response_latency_hrs": 12.0,
            "competitor_mentions": 0
        }
        rec3 = generate_recovery_recommendation(d3, risk_score=75.0, nearest_analogs=[])
        primary_risk_signal3 = rec3["ranked_risks"]["primary"]["factor_name"]
        self.assertEqual(primary_risk_signal3, "Unstable Custom Scope")
        self.assertEqual(rec3["root_cause"], "Scope Instability")
        self.assertIn("freeze", rec3["primary_action"].lower())

        # Test Scenario 4: Single-threaded highest -> Lack of Stakeholder Alignment root cause & multi-thread recommendation
        d4 = {
            "deal_id": "D-STK",
            "stage": "Proposal",
            "days_in_stage": 10,
            "stakeholder_count": 1,
            "deal_size": 400000,
            "response_latency_hrs": 12.0,
            "competitor_mentions": 0
        }
        rec4 = generate_recovery_recommendation(d4, risk_score=80.0, nearest_analogs=[])
        primary_risk_signal4 = rec4["ranked_risks"]["primary"]["factor_name"]
        self.assertEqual(primary_risk_signal4, "Single-Threaded Relationship")
        self.assertEqual(rec4["root_cause"], "Lack of Stakeholder Alignment")
        self.assertIn("multi-threaded", rec4["primary_action"].lower())

        # Test Scenario 5: Strict invariant check (primary_root_cause maps directly to primary_risk_signal)
        FACTOR_MAPPING = {
            "Extended Time in Stage": "Extended Time in Stage / Deal Stagnation",
            "Competitor Presence": "Competitive Pressure",
            "Unstable Custom Scope": "Scope Instability",
            "Scope Adjustment Recorded": "Scope Instability",
            "Single-Threaded Relationship": "Lack of Stakeholder Alignment",
            "Elevated Response Latency": "Buyer Disengagement",
            "Declining Buyer Sentiment": "Buyer Disengagement"
        }
        for d_test in [d1, d2, d3, d4]:
            r_test = generate_recovery_recommendation(d_test, risk_score=80.0, nearest_analogs=[])
            p_factor = r_test["ranked_risks"]["primary"]["factor_name"]
            expected_cause = FACTOR_MAPPING[p_factor]
            self.assertEqual(r_test["root_cause"], expected_cause)

    def test_15_why_not_wait_analog_duration_count_accuracy(self):
        """Automated regression tests verifying exact analog-duration count accuracy in WHY NOT WAIT narrative."""
        from recommendation_engine import generate_recovery_recommendation

        deal_base = {"deal_id": "TEST-WNW", "stage": "Closing", "days_in_stage": 100}

        # TEST 1: days = [34, 41, 20, 116, 52], threshold = 20 -> expected count = 5/5
        analogs1 = [
            {"deal_id": "A1", "outcome": "lost", "days_in_stage": 34},
            {"deal_id": "A2", "outcome": "stalled", "days_in_stage": 41},
            {"deal_id": "A3", "outcome": "lost", "days_in_stage": 20},
            {"deal_id": "A4", "outcome": "stalled", "days_in_stage": 116},
            {"deal_id": "A5", "outcome": "lost", "days_in_stage": 52}
        ]
        rec1 = generate_recovery_recommendation(deal_base, risk_score=80.0, nearest_analogs=analogs1)
        self.assertIn("5/5 spent at least 20 days in Closing", rec1["why_not_wait"])

        # TEST 2: Hyperion scenario: days = [143, 178, 127, 135, 112], threshold = 112 -> expected count = 5/5
        analogs2 = [
            {"deal_id": "A1", "outcome": "lost", "days_in_stage": 143},
            {"deal_id": "A2", "outcome": "stalled", "days_in_stage": 178},
            {"deal_id": "A3", "outcome": "lost", "days_in_stage": 127},
            {"deal_id": "A4", "outcome": "stalled", "days_in_stage": 135},
            {"deal_id": "A5", "outcome": "lost", "days_in_stage": 112}
        ]
        rec2 = generate_recovery_recommendation({"deal_id": "HYP", "stage": "Qualification", "days_in_stage": 150}, risk_score=85.0, nearest_analogs=analogs2)
        self.assertIn("5/5 spent at least 112 days in Qualification", rec2["why_not_wait"])

        # TEST 3: days = [48, 21, 90, 97, 3], threshold = 97 -> expected count = 1/5 (e.g. if threshold is 97)
        # Note: if minimum of stalled/lost is 3, threshold is 3 -> 5/5. If threshold forced to 97:
        analogs3 = [
            {"deal_id": "A1", "outcome": "won", "days_in_stage": 48},
            {"deal_id": "A2", "outcome": "won", "days_in_stage": 21},
            {"deal_id": "A3", "outcome": "won", "days_in_stage": 90},
            {"deal_id": "A4", "outcome": "lost", "days_in_stage": 97},
            {"deal_id": "A5", "outcome": "won", "days_in_stage": 3}
        ]
        rec3 = generate_recovery_recommendation(deal_base, risk_score=80.0, nearest_analogs=analogs3)
        self.assertIn("1/5 spent at least 97 days in Closing", rec3["why_not_wait"])

        # TEST 4: Exact boundary behavior: days = [20, 19, 20, 21, 5], threshold = 20 -> expected count = 3/5
        analogs4 = [
            {"deal_id": "A1", "outcome": "lost", "days_in_stage": 20},
            {"deal_id": "A2", "outcome": "won", "days_in_stage": 19},
            {"deal_id": "A3", "outcome": "stalled", "days_in_stage": 20},
            {"deal_id": "A4", "outcome": "won", "days_in_stage": 21},
            {"deal_id": "A5", "outcome": "won", "days_in_stage": 5}
        ]
        rec4 = generate_recovery_recommendation(deal_base, risk_score=80.0, nearest_analogs=analogs4)
        self.assertIn("3/5 spent at least 20 days in Closing", rec4["why_not_wait"])

        # TEST 5: Fewer than 5 analogs: days = [30, 40, 10] -> denominator = 3 (e.g. 2/3)
        analogs5 = [
            {"deal_id": "A1", "outcome": "lost", "days_in_stage": 30},
            {"deal_id": "A2", "outcome": "stalled", "days_in_stage": 40},
            {"deal_id": "A3", "outcome": "won", "days_in_stage": 10}
        ]
        rec5 = generate_recovery_recommendation(deal_base, risk_score=80.0, nearest_analogs=analogs5)
        self.assertIn("2/3 spent at least 30 days in Closing", rec5["why_not_wait"])

if __name__ == "__main__":
    unittest.main()
