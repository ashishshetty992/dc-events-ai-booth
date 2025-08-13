import json
import httpx
import re
from models import BoothRequest, AgentResponse
from utils import load_all_datasets, compute_booth_analytics
import logging

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

async def analyze_booth_request(request: BoothRequest) -> AgentResponse:
    try:
        datasets = load_all_datasets()
        competitors = datasets["competitors"]
        analytics = compute_booth_analytics(request, datasets)
        prompt = build_prompt(request, competitors, analytics)

        print("🔁 Sending prompt to Ollama...")
        print(prompt)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

        result = response.json()["response"]
        print("✅ AI Response (raw):", result)

        # Clean and extract only JSON
        raw = result.strip()
        if "```json" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].strip()

        if not raw.strip().startswith("{"):
            raise ValueError("Non-JSON response from model")

        clean_lines = [line for line in raw.splitlines() if not line.strip().startswith("//")]
        clean_json = "\n".join(clean_lines)
        clean_json = re.sub(r",\s*}", "}", clean_json)

        # Ensure proper JSON object closing
        if "}" in clean_json:
            clean_json = clean_json[:clean_json.rfind("}") + 1]

        parsed = json.loads(clean_json)

        required_keys = {"summary", "approval_recommendation", "competitor_analysis", "risk_flags"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Missing keys in AI response: {set(required_keys) - set(parsed.keys())}")

        return AgentResponse(**parsed, analytics=analytics)

    except Exception as e:
        print("❌ Error from Ollama or JSON parse:", e)
        return AgentResponse(
            summary="Unable to parse AI response.",
            approval_recommendation="Manual review needed.",
            competitor_analysis=None,
            risk_flags=["AI parse error or non-JSON response"],
            analytics=None
        )

async def chat_with_llama(question: str, context: dict) -> str:
    logging.info(f"[CHAT] question={question}, context={context}")

    booth = context.get("last_booth")
    analytics = context.get("last_analytics")

    booth_info = ""
    if booth:
        booth_data = booth.dict()
        booth_info = f"""
Company: {booth_data.get("company_name")}
Event: {booth_data.get("event_name")} on {booth_data.get("event_date")}
Booth Size: {booth_data.get("booth_size_sqm")} sqm
Budget: ₹{booth_data.get("budget_inr")}
Location: {booth_data.get("location")}
Industry: {booth_data.get("industry")}
Past Events: {', '.join(booth_data.get("past_events", []))}
        """

    analytics_info = ""
    if analytics:
        lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in analytics.items()]
        analytics_info = "\n".join(lines)

    prompt = f"""
You are a helpful assistant reviewing booth analytics and event requests.

Here is the user's follow-up question:
"{question}"

📝 Booth Request:
{booth_info.strip()}

📊 Key Analytics:
{analytics_info.strip()}

💡 Respond with a helpful answer based on this context. 
Your ENTIRE reply MUST be a valid JSON object like:
{{
  "answer": "Your response here"
}}

❌ Do not include markdown, commentary, or anything outside this JSON format.
"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )

    raw_response = response.json().get("response", "").strip()

    # Clean and extract just the JSON
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].strip()

    return raw_response

async def chat_with_design_assistant(message: str, current_design: dict, session_id: str) -> str:
    """AI assistant for booth design questions and suggestions with Saudi market expertise."""
    logging.info(f"[DESIGN CHAT] session={session_id}, message={message}")
    
    design_info = ""
    pricing_info = ""
    
    if current_design:
        dimensions = current_design.get("dimensions", {})
        elements = current_design.get("elements", [])
        colors = current_design.get("colors", {})
        pricing = current_design.get("pricing", {})
        branding = current_design.get("branding", {})
        
        design_info = f"""
Current Booth Design for {branding.get("companyName", "Your Company")}:
- Dimensions: {dimensions.get("width", 6)}m × {dimensions.get("depth", 6)}m × {dimensions.get("height", 3)}m
- Elements: {len(elements)} items ({', '.join([el.get("label", el.get("type", "unknown")) for el in elements[:3]])})
- Color Scheme: {colors.get("primary", "#3B82F6")} (primary), {colors.get("secondary", "#F3F4F6")} (secondary)
- Style: {current_design.get("style", "modern")}
        """
        
        if pricing:
            pricing_info = f"""
Current Pricing (Saudi Market):
- Total Budget: SAR {pricing.get("total", 0):,}
- Materials: SAR {pricing.get("materials", 0):,}
- Elements: SAR {pricing.get("elements", 0):,}
- Setup: SAR {pricing.get("setup", 0):,}
            """

    prompt = f"""
You are a Saudi Arabia booth design expert with deep knowledge of local market, culture, and exhibition standards.

User request: "{message}"
{design_info.strip()}
{pricing_info.strip()}

SAUDI MARKET EXPERTISE:
- All pricing in Saudi Riyals (SAR)
- Cultural considerations (prayer areas, modest design, Arabic signage)
- Local regulations and standards
- Vision 2030 alignment opportunities
- Regional preferences (Riyadh, Jeddah, NEOM styles)

RESPONSE RULES:
1. Keep responses under 100 words
2. Include SAR pricing when relevant
3. Mention Saudi-specific considerations
4. Use bullet points or tables when possible
5. Be culturally sensitive and professional

RESPONSE FORMAT for design reviews:
| Element | Status | Suggestion | Cost (SAR) |
|---------|--------|------------|------------|
| Layout | ✅ Good | Add majlis seating | 8,250 |
| Signage | ⚠️ Missing | Add Arabic text | 1,500 |
| Tech | 🚀 Excellent | Add AI demo area | 30,000 |

For pricing suggestions:
- Always use SAR currency
- Mention cost-effective Saudi suppliers
- Consider local labor costs
- Reference typical Saudi event budgets

For cultural suggestions:
- Prayer area considerations
- Arabic language signage
- Local color preferences
- Saudi hospitality elements (tea station, majlis)

Be professional, concise, and showcase deep Saudi market knowledge.
"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )

    raw_response = response.json().get("response", "").strip()
    return raw_response
    
def build_prompt(request: BoothRequest, competitors: list[dict], analytics: dict) -> str:
    top_competitors = sorted(
        [c for c in competitors if c.get("industry", "").lower() == request.industry.lower()],
        key=lambda x: x.get("conversion_rate", 0),
        reverse=True
    )[:5]

    # Extract key insights from enhanced analytics
    ml_approval_prob = analytics.get("ml_approval_probability", 0.5)
    ml_roi_prediction = analytics.get("ml_predicted_roi", 2.0)
    market_opportunity = analytics.get("market_opportunity_score", 0.5)
    strategic_recommendations = analytics.get("strategic_recommendations", [])
    vision_alignment = analytics.get("vision_2030_alignment", [])
    risk_factors = analytics.get("risk_factors", [])
    market_opportunities = analytics.get("market_opportunities", [])
    
    return f"""
You are an advanced AI decision agent for AddEnterprise, specialized in Saudi Arabian market analysis and booth approval decisions.

🎯 Your response MUST be a single, valid JSON object with exactly these keys:

{{
  "summary": "string",
  "approval_recommendation": "Approve | Reject | Needs Review",
  "competitor_analysis": "string",
  "risk_flags": ["string", "string", ...]
}}

🇸🇦 SAUDI ARABIA CONTEXT:
- Vision 2030 transformation priorities
- NEOM and giga-projects development
- Economic diversification initiatives
- Strong government support for key industries
- Regulatory environment and compliance requirements

📊 ENHANCED ANALYTICS INSIGHTS:
- ML Approval Probability: {ml_approval_prob:.1%}
- ML ROI Prediction: {ml_roi_prediction:.1f}x
- Market Opportunity Score: {market_opportunity:.1%}
- Competition Intensity: {analytics.get("competition_intensity", "Medium")}
- Regulatory Complexity: {analytics.get("regulatory_complexity", "Medium")}

🎯 STRATEGIC RECOMMENDATIONS:
{chr(10).join([f"• {rec}" for rec in strategic_recommendations[:5]])}

🌟 VISION 2030 ALIGNMENT:
{chr(10).join([f"• {align}" for align in vision_alignment[:3]])}

⚠️ MARKET RISKS:
{chr(10).join([f"• {risk}" for risk in risk_factors[:3]])}

🚀 MARKET OPPORTUNITIES:
{chr(10).join([f"• {opp}" for opp in market_opportunities[:3]])}

📌 SUMMARY REQUIREMENTS:
- Reference ML predictions and market opportunity score
- Mention Vision 2030 alignment if applicable
- Include regulatory complexity assessment
- Highlight key strategic recommendations

📌 COMPETITOR ANALYSIS REQUIREMENTS:
- Compare against top Saudi competitors in the industry
- Reference market share, digital maturity, and sustainability scores
- Mention government relations and local partnerships

📌 APPROVAL DECISION LOGIC:
- ML Approval Probability > 75% + High Market Opportunity → "Approve"
- ML Approval Probability < 40% OR Critical Risks → "Reject"  
- Regulatory complexity "High" + Multiple risks → "Needs Review"
- Vision 2030 alignment + Strategic value → Weight toward "Approve"

📌 RISK FLAGS PRIORITIES:
- Regulatory compliance issues
- Market opportunity limitations
- ML-identified risks
- Competition intensity concerns
- ROI predictions below industry standards

🔍 Booth Request:
{request.json()}

📊 Complete Analytics:
{json.dumps(analytics, indent=2)}

📁 Top Saudi Competitors in {request.industry}:
{json.dumps(top_competitors, indent=2)}

🚫 DO NOT include markdown, bullet points, prose, or extra commentary.
✅ DO ONLY return a valid JSON object as output.
"""