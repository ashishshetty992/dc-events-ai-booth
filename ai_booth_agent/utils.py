import os
import json
import numpy as np
from typing import Dict
from models import BoothRequest
from advanced_analytics import SaudiMarketAnalyzer

# Set your local 'data/' directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASETS = {
    "competitors": "enhanced_competitor_data_saudi.json",
    "event_schedule": "enhanced_event_schedule_saudi.json", 
    "global_trends": "global_vendor_trends.json",
    "vendor_outcomes": "vendor_event_outcomes.json",
    "audience_profiles": "audience_profiles.json",
    "approval_policy": "approval_policy.json",
    "resource_calendar": "resource_calendar.json",
    "market_intelligence": "saudi_market_intelligence.json",
    "regulations": "saudi_business_regulations.json"
}

# Initialize advanced analytics engine
_analytics_engine = None

def get_analytics_engine():
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = SaudiMarketAnalyzer(DATA_DIR)
    return _analytics_engine


def load_all_datasets() -> Dict[str, list]:
    loaded_data = {}
    for key, filename in DATASETS.items():
        try:
            with open(os.path.join(DATA_DIR, filename), "r") as f:
                loaded_data[key] = json.load(f)
        except Exception as e:
            print(f"❌ Could not load {filename}: {e}")
            loaded_data[key] = []
    return loaded_data


def compute_booth_analytics(request: BoothRequest, datasets: dict) -> dict:
    # 1. Competitor benchmarking
    competitors = [
        c for c in datasets["competitors"]
        if c["industry"].lower() == request.industry.lower()
        and c["region"].lower() == "saudi"
    ]

    booth_efficiency = request.budget_inr / request.booth_size_sqm
    competitor_efficiencies = [c["budget"] / c["booth_size"] for c in competitors if c["booth_size"] > 0]
    median_eff = np.median(competitor_efficiencies) if competitor_efficiencies else 0

    size_list = [c["booth_size"] for c in competitors]
    percentile_size = np.percentile(size_list, np.searchsorted(sorted(size_list), request.booth_size_sqm)) if size_list else 0

    # 2. Global industry trend
    global_industry = next((g for g in datasets["global_trends"] if g["industry"].lower() == request.industry.lower()), {})
    growth_score = global_industry.get("growth_score", 0)
    avg_global_booth = global_industry.get("avg_booth_size", 0)
    conversion_ceiling = global_industry.get("conversion_ceiling", 0)

    # 3. Event match
    event_match = next((e for e in datasets["event_schedule"]
                        if request.event_name and request.event_name.lower() in e["event_name"].lower()), None)

    # 4. Audience alignment
    audience_match_score = None
    if event_match:
        ap = next((a for a in datasets["audience_profiles"]
                   if a["event_name"].lower() in event_match["event_name"].lower()), None)
        if ap:
            audience_match_score = ap.get("match_score")

    # 5. Past performance (ROI)
    past_perf = [p for p in datasets["vendor_outcomes"] if p["company"].lower() == request.company_name.lower()]
    avg_roi = np.mean([p["roi_score"] for p in past_perf]) if past_perf else None
    flagged_issues = [i for p in past_perf for i in p.get("issues_flagged", [])]

    # 6. Approval policy rule
    matched_rule = None
    for rule in datasets["approval_policy"]:
        try:
            if eval(rule["condition"], {}, {
                "booth_size": request.booth_size_sqm,
                "budget": request.budget_inr,
                "industry": request.industry
            }):
                matched_rule = rule
                break
        except Exception as e:
            print(f"⚠️ Policy rule eval failed: {e}")
            continue

    # 7. Logistics check
    logistic_conflict = False
    if request.event_date and event_match:
        request_city = event_match.get("location", "").lower()
        for r in datasets["resource_calendar"]:
            if r["date"] == request.event_date and r["city"].lower() in request_city:
                logistic_conflict = r["available_crews"] <= 1
                break

    # Get advanced analytics
    analytics_engine = get_analytics_engine()
    
    # ML-based predictions
    approval_probability = analytics_engine.predict_approval_probability(request.dict())
    predicted_roi = analytics_engine.predict_roi(request.dict())
    
    # Market opportunity analysis
    market_analysis = analytics_engine.analyze_market_opportunity(
        request.industry, 
        request.location or "Riyadh"
    )
    
    # Strategic recommendations
    basic_analytics = {
        "booth_efficiency": round(booth_efficiency, 2),
        "industry_median_efficiency": round(median_eff, 2),
        "booth_size_percentile": round(percentile_size, 2),
        "global_growth_score": growth_score,
        "global_avg_booth_size": avg_global_booth,
        "global_conversion_ceiling": conversion_ceiling,
        "past_avg_roi": round(avg_roi, 2) if avg_roi else None,
        "issues_flagged": flagged_issues if flagged_issues else [],
        "audience_match_score": round(audience_match_score, 2) if audience_match_score else None,
        "event_match": event_match["event_name"] if event_match else "Unlisted",
        "logistics_conflict": logistic_conflict,
        "matched_approval_rule": matched_rule["rule_name"] if matched_rule else None,
        "suggested_action": matched_rule["action"] if matched_rule else "Review"
    }
    
    strategic_recommendations = analytics_engine.generate_strategic_recommendations(
        request.dict(), 
        basic_analytics
    )
    
    # Enhanced analytics with ML predictions and market intelligence
    enhanced_analytics = {
        **basic_analytics,
        "ml_approval_probability": round(approval_probability, 3),
        "ml_predicted_roi": round(predicted_roi, 2),
        "market_opportunity_score": round(market_analysis["opportunity_score"], 3),
        "market_size_usd": market_analysis.get("market_size_usd", 0),
        "competition_intensity": market_analysis.get("competition_intensity", "Medium"),
        "regulatory_complexity": market_analysis.get("regulatory_complexity", "Medium"),
        "vision_2030_alignment": market_analysis.get("vision_2030_alignment", []),
        "strategic_recommendations": strategic_recommendations,
        "risk_factors": market_analysis.get("risk_factors", []),
        "market_opportunities": market_analysis.get("opportunities", []),
        "confidence_score": round((approval_probability + market_analysis["opportunity_score"]) / 2, 3)
    }
    
    return enhanced_analytics