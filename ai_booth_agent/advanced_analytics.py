"""
Advanced Analytics Engine for Saudi Arabia Booth Approval System
Implements machine learning models and sophisticated analytics
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SaudiMarketAnalyzer:
    """Advanced market intelligence and predictive analytics for Saudi market"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.market_intelligence = self._load_market_intelligence()
        self.competitors = self._load_enhanced_competitors()
        self.regulations = self._load_regulations()
        self.events = self._load_events()
        
        # Initialize ML models
        self.approval_classifier = None
        self.roi_predictor = None
        self.market_segmenter = None
        self._train_models()
    
    def _load_market_intelligence(self) -> Dict:
        """Load comprehensive Saudi market intelligence"""
        try:
            with open(os.path.join(self.data_dir, "saudi_market_intelligence.json"), "r") as f:
                return json.load(f)
        except:
            return {}
    
    def _load_enhanced_competitors(self) -> List[Dict]:
        """Load enhanced competitor data"""
        try:
            with open(os.path.join(self.data_dir, "enhanced_competitor_data_saudi.json"), "r") as f:
                return json.load(f)
        except:
            return []
    
    def _load_regulations(self) -> Dict:
        """Load Saudi business regulations"""
        try:
            with open(os.path.join(self.data_dir, "saudi_business_regulations.json"), "r") as f:
                return json.load(f)
        except:
            return {}
    
    def _load_events(self) -> List[Dict]:
        """Load enhanced event schedule"""
        try:
            with open(os.path.join(self.data_dir, "enhanced_event_schedule_saudi.json"), "r") as f:
                return json.load(f)
        except:
            return []
    
    def _train_models(self):
        """Train machine learning models for predictions"""
        if not self.competitors:
            return
        
        # Prepare training data
        df = pd.DataFrame(self.competitors)
        
        # Features for approval prediction
        features = ['booth_size', 'budget', 'employees', 'market_share', 
                   'satisfaction_score', 'lead_quality', 'brand_recognition',
                   'digital_maturity', 'sustainability_score', 'innovation_index',
                   'government_relations', 'local_partnerships', 'compliance_score']
        
        # Create approval labels (simplified logic)
        df['approval'] = ((df['compliance_score'] > 0.8) & 
                         (df['satisfaction_score'] > 4.0) & 
                         (df['sustainability_score'] > 0.7)).astype(int)
        
        X = df[features].fillna(0)
        y_approval = df['approval']
        y_roi = df['roi_3_year']
        
        # Train approval classifier
        self.approval_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.approval_classifier.fit(X, y_approval)
        
        # Train ROI predictor
        self.roi_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.roi_predictor.fit(X, y_roi)
        
        # Train market segmentation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.market_segmenter = KMeans(n_clusters=5, random_state=42)
        self.market_segmenter.fit(X_scaled)
        
        self.scaler = scaler
    
    def predict_approval_probability(self, booth_request: Dict) -> float:
        """Predict approval probability using ML model"""
        if not self.approval_classifier:
            return 0.5
        
        # Extract features from request
        features = self._extract_request_features(booth_request)
        features_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Predict probability
        prob = self.approval_classifier.predict_proba(features_array)[0, 1]
        return float(prob)
    
    def predict_roi(self, booth_request: Dict) -> float:
        """Predict expected ROI using ML model"""
        if not self.roi_predictor:
            return 2.0
        
        features = self._extract_request_features(booth_request)
        features_array = np.array([list(features.values())]).reshape(1, -1)
        
        roi = self.roi_predictor.predict(features_array)[0]
        return max(0.5, float(roi))  # Minimum ROI of 0.5
    
    def _extract_request_features(self, booth_request: Dict) -> Dict:
        """Extract ML features from booth request"""
        industry = booth_request.get('industry', 'Other')
        
        # Get industry averages from competitors
        industry_competitors = [c for c in self.competitors if c['industry'] == industry]
        
        if industry_competitors:
            avg_satisfaction = np.mean([c['satisfaction_score'] for c in industry_competitors])
            avg_lead_quality = np.mean([c['lead_quality'] for c in industry_competitors])
            avg_brand_recognition = np.mean([c['brand_recognition'] for c in industry_competitors])
            avg_digital_maturity = np.mean([c['digital_maturity'] for c in industry_competitors])
            avg_sustainability = np.mean([c['sustainability_score'] for c in industry_competitors])
            avg_innovation = np.mean([c['innovation_index'] for c in industry_competitors])
            avg_gov_relations = np.mean([c['government_relations'] for c in industry_competitors])
            avg_partnerships = np.mean([c['local_partnerships'] for c in industry_competitors])
            avg_compliance = np.mean([c['compliance_score'] for c in industry_competitors])
        else:
            # Default values
            avg_satisfaction = 4.0
            avg_lead_quality = 0.8
            avg_brand_recognition = 0.7
            avg_digital_maturity = 0.75
            avg_sustainability = 0.8
            avg_innovation = 0.75
            avg_gov_relations = 0.8
            avg_partnerships = 10
            avg_compliance = 0.9
        
        # Estimate company features based on budget and size
        budget = booth_request.get('budget_inr', 0)
        booth_size = booth_request.get('booth_size_sqm', 0)
        
        # Estimate employees based on budget (rough heuristic)
        estimated_employees = min(max(budget / 1000, 10), 50000)
        
        # Estimate market share (smaller companies typically have lower share)
        estimated_market_share = min(budget / 10000000 * 5, 25)
        
        return {
            'booth_size': booth_size,
            'budget': budget,
            'employees': estimated_employees,
            'market_share': estimated_market_share,
            'satisfaction_score': avg_satisfaction,
            'lead_quality': avg_lead_quality,
            'brand_recognition': avg_brand_recognition,
            'digital_maturity': avg_digital_maturity,
            'sustainability_score': avg_sustainability,
            'innovation_index': avg_innovation,
            'government_relations': avg_gov_relations,
            'local_partnerships': avg_partnerships,
            'compliance_score': avg_compliance
        }
    
    def analyze_market_opportunity(self, industry: str, location: str) -> Dict:
        """Analyze market opportunity for specific industry and location"""
        opportunity_score = 0.5
        risk_factors = []
        opportunities = []
        
        # Industry growth analysis
        if industry in self.market_intelligence.get('industry_growth_forecasts', {}):
            industry_data = self.market_intelligence['industry_growth_forecasts'][industry]
            growth_rate = industry_data.get('growth_rate_2025', 5.0)
            
            if growth_rate > 15:
                opportunity_score += 0.2
                opportunities.append(f"High industry growth rate: {growth_rate}%")
            elif growth_rate > 10:
                opportunity_score += 0.1
                opportunities.append(f"Good industry growth rate: {growth_rate}%")
            elif growth_rate < 5:
                opportunity_score -= 0.1
                risk_factors.append(f"Low industry growth rate: {growth_rate}%")
            
            # Government support analysis
            govt_spending = industry_data.get('government_spending_usd', 0)
            if govt_spending > 10000000000:
                opportunity_score += 0.15
                opportunities.append("High government investment in sector")
            
            # Talent gap analysis
            talent_gap = industry_data.get('talent_gap', 0.3)
            if talent_gap > 0.4:
                risk_factors.append("High talent shortage in industry")
                opportunity_score -= 0.1
            
            # Regulatory support
            reg_support = industry_data.get('regulatory_support', 0.8)
            if reg_support > 0.85:
                opportunity_score += 0.1
                opportunities.append("Strong regulatory support")
        
        # Regional analysis
        if location in self.market_intelligence.get('regional_insights', {}):
            regional_data = self.market_intelligence['regional_insights'][location]
            
            # Business density
            business_density = regional_data.get('business_density', 0.7)
            if business_density > 0.85:
                opportunity_score += 0.1
                opportunities.append("High business concentration in region")
            
            # Infrastructure score
            infrastructure = regional_data.get('infrastructure_score', 0.8)
            if infrastructure > 0.9:
                opportunity_score += 0.1
                opportunities.append("Excellent infrastructure")
            elif infrastructure < 0.7:
                risk_factors.append("Infrastructure challenges")
                opportunity_score -= 0.1
            
            # Government procurement share
            govt_procurement = regional_data.get('government_procurement_share', 0.3)
            if govt_procurement > 0.4:
                opportunity_score += 0.1
                opportunities.append("High government procurement activity")
        
        # Vision 2030 alignment
        vision_priorities = self.market_intelligence.get('vision_2030_priorities', {})
        relevant_priorities = []
        
        if industry in ['IT & Technology', 'Energy', 'Tourism', 'Manufacturing']:
            for priority, progress in vision_priorities.items():
                if industry.lower() in priority.lower() or any(word in priority.lower() for word in industry.lower().split()):
                    if progress > 0.7:
                        opportunity_score += 0.1
                        relevant_priorities.append(f"{priority}: {progress:.0%} progress")
        
        return {
            'opportunity_score': min(1.0, max(0.0, opportunity_score)),
            'risk_factors': risk_factors,
            'opportunities': opportunities,
            'vision_2030_alignment': relevant_priorities,
            'market_size_usd': self.market_intelligence.get('industry_growth_forecasts', {}).get(industry, {}).get('market_size_usd', 0),
            'competition_intensity': self._calculate_competition_intensity(industry),
            'regulatory_complexity': self._calculate_regulatory_complexity(industry)
        }
    
    def _calculate_competition_intensity(self, industry: str) -> str:
        """Calculate competition intensity in the industry"""
        industry_competitors = [c for c in self.competitors if c['industry'] == industry]
        
        if len(industry_competitors) < 3:
            return "Low"
        elif len(industry_competitors) < 7:
            return "Moderate" 
        else:
            return "High"
    
    def _calculate_regulatory_complexity(self, industry: str) -> str:
        """Calculate regulatory complexity for the industry"""
        if not self.regulations:
            return "Medium"
        
        complexity_score = 0
        
        # Check various regulatory requirements
        investment_reqs = self.regulations.get('investment_requirements', {})
        compliance_reqs = self.regulations.get('compliance_requirements', {})
        
        if investment_reqs.get('local_partner_required', {}).get(industry, False):
            complexity_score += 1
        
        if investment_reqs.get('government_approval_required', {}).get(industry, False):
            complexity_score += 1
        
        saudization_quota = compliance_reqs.get('saudization_quotas', {}).get(industry, 0.3)
        if saudization_quota > 0.6:
            complexity_score += 1
        
        env_compliance = compliance_reqs.get('environmental_compliance', {}).get(industry, 0.7)
        if env_compliance > 0.9:
            complexity_score += 1
        
        if complexity_score <= 1:
            return "Low"
        elif complexity_score <= 2:
            return "Medium"
        else:
            return "High"
    
    def generate_strategic_recommendations(self, booth_request: Dict, analytics: Dict) -> List[str]:
        """Generate strategic recommendations based on comprehensive analysis"""
        recommendations = []
        industry = booth_request.get('industry', '')
        budget = booth_request.get('budget_inr', 0)
        booth_size = booth_request.get('booth_size_sqm', 0)
        location = booth_request.get('location', '')
        
        # ML-based approval probability
        approval_prob = self.predict_approval_probability(booth_request)
        predicted_roi = self.predict_roi(booth_request)
        
        if approval_prob > 0.8:
            recommendations.append(f"🚀 High approval probability ({approval_prob:.0%}) - Strong application")
        elif approval_prob < 0.4:
            recommendations.append(f"⚠️ Low approval probability ({approval_prob:.0%}) - Consider optimization")
        
        if predicted_roi > 3.0:
            recommendations.append(f"💰 Excellent ROI potential ({predicted_roi:.1f}x) - High value opportunity")
        elif predicted_roi < 1.5:
            recommendations.append(f"📉 Below average ROI prediction ({predicted_roi:.1f}x) - Review cost structure")
        
        # Market opportunity analysis
        market_analysis = self.analyze_market_opportunity(industry, location)
        opportunity_score = market_analysis['opportunity_score']
        
        if opportunity_score > 0.7:
            recommendations.append(f"🎯 Excellent market opportunity (score: {opportunity_score:.0%})")
            recommendations.extend([f"• {opp}" for opp in market_analysis['opportunities'][:2]])
        
        if market_analysis['risk_factors']:
            recommendations.append("⚠️ Market risks to consider:")
            recommendations.extend([f"• {risk}" for risk in market_analysis['risk_factors'][:2]])
        
        # Budget optimization recommendations
        efficiency = budget / booth_size if booth_size > 0 else 0
        industry_competitors = [c for c in self.competitors if c['industry'] == industry]
        
        if industry_competitors:
            avg_efficiency = np.mean([c['budget'] / c['booth_size'] for c in industry_competitors if c['booth_size'] > 0])
            
            if efficiency > avg_efficiency * 1.5:
                recommendations.append(f"💸 Budget efficiency below industry average - consider optimizing cost per sqm")
            elif efficiency < avg_efficiency * 0.7:
                recommendations.append(f"💎 Excellent budget efficiency - well optimized spend")
        
        # Regulatory compliance recommendations
        if industry in self.regulations.get('investment_requirements', {}).get('local_partner_required', {}):
            if self.regulations['investment_requirements']['local_partner_required'][industry]:
                recommendations.append("🤝 Local partnership required - ensure compliance documentation")
        
        if industry in self.regulations.get('compliance_requirements', {}).get('saudization_quotas', {}):
            quota = self.regulations['compliance_requirements']['saudization_quotas'][industry]
            if quota > 0.5:
                recommendations.append(f"👥 High Saudization quota required ({quota:.0%}) - plan staffing accordingly")
        
        # Event-specific recommendations
        event_name = booth_request.get('event_name', '')
        matching_events = [e for e in self.events if event_name.lower() in e['event_name'].lower()]
        
        if matching_events:
            event = matching_events[0]
            if event.get('international_attendees_pct', 0) > 0.6:
                recommendations.append("🌍 High international attendance - prepare multilingual materials")
            
            if event.get('government_presence', 0) > 0.8:
                recommendations.append("🏛️ Strong government presence - opportunity for policy engagement")
            
            if event.get('lead_quality_index', 0) > 0.85:
                recommendations.append("⭐ High-quality lead event - focus on conversion strategies")
        
        # Vision 2030 alignment
        vision_alignment = market_analysis.get('vision_2030_alignment', [])
        if vision_alignment:
            recommendations.append("🇸🇦 Strong Vision 2030 alignment:")
            recommendations.extend([f"• {alignment}" for alignment in vision_alignment[:2]])
        
        return recommendations[:8]  # Limit to top 8 recommendations

def create_advanced_analytics_instance(data_dir: str) -> SaudiMarketAnalyzer:
    """Create and return an instance of the advanced analytics engine"""
    return SaudiMarketAnalyzer(data_dir)