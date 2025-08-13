"""
FastAPI application for AI Booth Approval Agent.

This module provides REST and WebSocket endpoints for booth request analysis,
chat functionality, and analytics data retrieval.
"""

import asyncio
import json
import logging
import os
import re
import statistics
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from agent_logic import analyze_booth_request, chat_with_llama, chat_with_design_assistant
from db import SessionContext, ChatHistory, BoothDesign, ApprovalDecision, SessionLocal, init_db
from models import AgentResponse, BoothRequest, BoothDesignModel, BoothDesignRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration."""
    # Allow both development and production origins
    CORS_ORIGINS = [
        "http://localhost:8080",
        "http://localhost:8081", 
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:5173",
        # Add Vercel deployment patterns
        "https://*.vercel.app",
        "https://vercel.app"
    ]
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    SYSTEM_PROMPT = (
        "You are an AI assistant. For every answer, return ONLY a valid JSON object with keys: "
        "summary, approval_recommendation, competitor_analysis, risk_flags, analytics. "
        "Do not include any text, markdown, or commentary outside the JSON."
    )

# Initialize FastAPI app
app = FastAPI(
    title="AI Booth Approval Agent",
    description="AI-powered booth approval system with chat and analytics",
    version="1.0.0"
)

# Initialize database
init_db()

# Add CORS middleware with environment-aware configuration
cors_origins = Config.CORS_ORIGINS.copy()

# In production, allow all origins (Vercel deployments)
# In development, use specific origins
import os
if os.getenv('VERCEL') or os.getenv('NODE_ENV') == 'production':
    cors_origins = ["*"]  # Allow all origins in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db() -> Session:
    """Get database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions
class ChatManager:
    """Manages chat operations and database interactions."""
    
    @staticmethod
    def save_chat(db: Session, session_id: str, role: str, message: str) -> None:
        """Save a chat message to the database."""
        try:
            db.add(ChatHistory(
                session_id=session_id,
                role=role,
                message=message,
                timestamp=datetime.utcnow()
            ))
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save chat: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def load_session_context(db: Session, session_id: str) -> Dict[str, Any]:
        """Load session context from database."""
        context = {}
        try:
            session_ctx = db.query(SessionContext).filter_by(session_id=session_id).first()
            if session_ctx:
                if session_ctx.last_booth:
                    context["last_booth"] = BoothRequest.parse_raw(session_ctx.last_booth)
                if session_ctx.last_analytics:
                    context["last_analytics"] = session_ctx.last_analytics
                logger.info(f"Loaded context for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to load session context: {e}")
        return context
    
    @staticmethod
    def save_session_context(
        db: Session, 
        session_id: str, 
        booth_request: BoothRequest, 
        analytics: Dict[str, Any]
    ) -> None:
        """Save session context to database."""
        try:
            booth_json = booth_request.json()
            ctx = db.query(SessionContext).filter_by(session_id=session_id).first()
            if ctx:
                ctx.last_booth = booth_json
                ctx.last_analytics = analytics
                ctx.updated_at = datetime.utcnow()
            else:
                ctx = SessionContext(
                    session_id=session_id,
                    last_booth=booth_json,
                    last_analytics=analytics
                )
                db.add(ctx)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save session context: {e}")
            db.rollback()
            raise

class AnalyticsManager:
    """Manages analytics data operations."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_json_data(filename: str) -> Dict[str, Any]:
        """Load JSON data from file with caching."""
        data_path = os.path.join(Config.DATA_DIR, filename)
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data file {filename}: {e}")
            return {}
    
    @staticmethod
    def analyze_competitor_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitor data and generate insights."""
        if not data:
            return {"data": [], "analytics": {}, "ai_insights": []}
        
        top_conversion = max(data, key=lambda x: x.get("conversion_rate", 0))
        top_footfall = max(data, key=lambda x: x.get("avg_footfall", 0))
        avg_budget = statistics.mean([x["budget"] for x in data])
        high_budget = [x for x in data if x["budget"] > avg_budget * 1.5]
        
        ai_insights = [
            f"Top conversion: {top_conversion['company']} ({top_conversion['conversion_rate']*100:.1f}%)",
            f"Top footfall: {top_footfall['company']} ({top_footfall['avg_footfall']})",
            f"{len(high_budget)} companies have budgets much higher than average."
        ]
        
        return {
            "data": data,
            "analytics": {"avg_budget": avg_budget},
            "ai_insights": ai_insights
        }
    
    @staticmethod
    def analyze_vendor_outcomes(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze vendor outcomes data and generate insights."""
        if not data:
            return {"data": [], "analytics": {}, "ai_insights": []}
        
        top_roi = max(data, key=lambda x: x.get("roi_score", 0))
        low_satisfaction = min(data, key=lambda x: x.get("satisfaction", 5))
        issues = [x for x in data if x.get("issues_flagged")]
        
        ai_insights = [
            f"Best ROI: {top_roi['company']} at {top_roi['event_name']} ({top_roi['roi_score']})",
            f"Lowest satisfaction: {low_satisfaction['company']} ({low_satisfaction['satisfaction']}/5)",
            f"{len(issues)} events had issues flagged."
        ]
        
        return {"data": data, "analytics": {}, "ai_insights": ai_insights}

class ResponseHandler:
    """Handles response formatting and JSON parsing."""
    
    @staticmethod
    def safe_json_parse(ai_response: str) -> Dict[str, Any]:
        """Safely parse AI response as JSON with fallback fixes."""
        try:
            return json.loads(ai_response)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = ai_response.strip()
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',([\s\n]*[}\]])', r'\1', fixed)
            # Add missing closing brace if needed
            if not fixed.endswith('}'):
                fixed += '}'
            try:
                return json.loads(fixed)
            except Exception:
                logger.error(f"Failed to parse AI response as JSON: {ai_response}")
                return {"error": "Invalid JSON from AI", "raw": ai_response}
    
    @staticmethod
    def format_booth_response(response: AgentResponse, session_id: str) -> Dict[str, Any]:
        """Format booth analysis response."""
        return {
            "session_id": session_id,
            "message": {
                "summary": response.summary,
                "recommendation": response.approval_recommendation,
                "insights": response.competitor_analysis,
                "risks": response.risk_flags,
                "analytics": response.analytics
            }
        }
    
    @staticmethod
    def format_chat_response(parsed_response: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Format chat response."""
        return {
            "session_id": session_id,
            "message": parsed_response
        }
    
    @staticmethod
    def format_error_response(session_id: str, error: str, details: str = "") -> Dict[str, Any]:
        """Format error response."""
        return {
            "session_id": session_id,
            "error": error,
            "details": details
        }

# REST Endpoints
@app.post("/api/chat/save")
async def save_chat_message(
    chat_data: dict,
    db: Session = Depends(get_db)
):
    """Save a chat message to the database."""
    try:
        session_id = chat_data.get("session_id")
        role = chat_data.get("role", "user")
        message = chat_data.get("message", "")
        
        if not session_id or not message:
            raise HTTPException(status_code=400, detail="session_id and message are required")
        
        ChatManager.save_chat(db, session_id, role, message)
        return {"status": "success", "message": "Chat saved successfully"}
        
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save chat: {str(e)}")

@app.post("/api/booth-design/save")
async def save_booth_design(
    design_data: dict,
    db: Session = Depends(get_db)
):
    """Save a booth design to the database."""
    try:
        session_id = design_data.get("session_id")
        design_name = design_data.get("design_name", "Untitled Design")
        design_json = design_data.get("design_data", {})
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        # Check if design already exists for this session
        existing_design = db.query(BoothDesign).filter_by(session_id=session_id).first()
        
        if existing_design:
            existing_design.design_name = design_name
            existing_design.design_data = design_json
            existing_design.updated_at = datetime.utcnow()
        else:
            new_design = BoothDesign(
                session_id=session_id,
                design_name=design_name,
                design_data=design_json
            )
            db.add(new_design)
        
        db.commit()
        return {"status": "success", "message": "Design saved successfully"}
        
    except Exception as e:
        logger.error(f"Error saving design: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save design: {str(e)}")

@app.get("/api/booth-design/load/{session_id}")
async def load_booth_design(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Load a booth design from the database."""
    try:
        design = db.query(BoothDesign).filter_by(session_id=session_id).first()
        
        if design:
            return {
                "status": "success",
                "design_name": design.design_name,
                "design_data": design.design_data,
                "updated_at": design.updated_at.isoformat() if design.updated_at else None
            }
        else:
            return {"status": "not_found", "message": "No design found for this session"}
        
    except Exception as e:
        logger.error(f"Error loading design: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load design: {str(e)}")

@app.post("/api/booth-design/chat")
async def booth_design_chat(
    chat_request: dict,
    db: Session = Depends(get_db)
):
    """Handle booth design chat with AI intelligence."""
    try:
        session_id = chat_request.get("session_id")
        message = chat_request.get("message", "")
        current_design = chat_request.get("current_design", {})
        
        if not session_id or not message:
            raise HTTPException(status_code=400, detail="session_id and message are required")
        
        logger.info(f"Processing design chat for session {session_id}: {message[:100]}...")
        
        # Get AI response with design context
        ai_response = await chat_with_design_assistant(message, current_design, session_id)
        
        return {
            "status": "success",
            "response": ai_response,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in design chat: {e}")
        return {
            "status": "error",
            "response": "I apologize, but I'm having trouble processing your request right now. Please ensure your message is clear and try again. If the problem persists, the AI service might be temporarily unavailable.",
            "error": str(e),
            "session_id": session_id if 'session_id' in locals() else None
        }

@app.post("/api/finalize-design")
async def finalize_design(
    report_data: dict,
    db: Session = Depends(get_db)
):
    """Finalize a booth design with screenshot and generate final report."""
    try:
        session_id = report_data.get("session_id")
        design_name = report_data.get("design_name", "Untitled Design")
        design_data = report_data.get("design_data", {})
        screenshot = report_data.get("screenshot", "")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        # Generate unique report ID
        import uuid
        report_id = str(uuid.uuid4())
        
        # Create comprehensive final report
        final_report = {
            "report_id": report_id,
            "session_id": session_id,
            "design_name": design_name,
            "design_data": design_data,
            "screenshot": screenshot,
            "timestamp": report_data.get("timestamp"),
            "pricing": report_data.get("pricing", {}),
            "elements_count": report_data.get("elements_count", 0),
            "booth_dimensions": report_data.get("booth_dimensions", ""),
            "total_cost_sar": report_data.get("total_cost_sar", 0),
            "status": "finalized",
            "approval_status": "pending_review",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Save to database (you might want to create a FinalReports table)
        # For now, we'll save it as a special booth design entry
        finalized_design = BoothDesign(
            session_id=f"finalized_{session_id}",
            design_name=f"FINAL_REPORT_{design_name}",
            design_data=final_report
        )
        db.add(finalized_design)
        db.commit()
        
        # Log the finalization
        logger.info(f"Design finalized for session {session_id}, report ID: {report_id}")
        
        return {
            "status": "success",
            "message": "Design finalized successfully",
            "report_id": report_id,
            "approval_status": "pending_review",
            "next_steps": [
                "Report generated and saved",
                "Submitted for management approval",
                "Screenshots archived",
                "Cost analysis completed"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error finalizing design: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to finalize design: {str(e)}")

@app.post("/submit_booth_request", response_model=AgentResponse)
async def submit_booth_request(
    request: BoothRequest,
    db: Session = Depends(get_db),
    session_id: str = Query(default_factory=lambda: str(uuid.uuid4()))
) -> AgentResponse:
    """Handle booth request submission via POST."""
    logger.info(f"Processing booth request for session {session_id}")
    
    try:
        # Save user request
        ChatManager.save_chat(db, session_id, "user", request.json())
        
        # Analyze request
        response = await analyze_booth_request(request)
        
        # Save bot response
        ChatManager.save_chat(db, session_id, "bot", response.model_dump_json())
        
        logger.info(f"Completed booth analysis for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing booth request for session {session_id}: {e}")
        raise

@app.get("/latest_booth_request")
async def get_latest_booth_request(db: Session = Depends(get_db)):
    """Get the most recent booth request submission."""
    try:
        # Get the most recent user message that contains booth request data
        latest_chat = db.query(ChatHistory).filter(
            ChatHistory.role == "user"
        ).order_by(ChatHistory.timestamp.desc()).first()
        
        if latest_chat and latest_chat.message:
            try:
                # Parse the JSON message to get booth request data
                request_data = json.loads(latest_chat.message)
                logger.info(f"Retrieved latest booth request: {request_data}")
                return request_data
            except json.JSONDecodeError:
                logger.warning("Latest chat message is not valid JSON")
                return None
        
        logger.info("No booth request found")
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving latest booth request: {e}")
        return None

# WebSocket Endpoint
@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Handle WebSocket chat connections."""
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    logger.info(f"WebSocket connected: {session_id}")
    
    # Load session context
    db = SessionLocal()
    context = ChatManager.load_session_context(db, session_id)
    db.close()
    
    try:
        while True:
            incoming = await websocket.receive_text()
            logger.info(f"Received message from {session_id}: {incoming[:100]}...")
            
            db = SessionLocal()
            try:
                await handle_websocket_message(websocket, db, session_id, incoming, context)
            except Exception as e:
                logger.error(f"Error handling WebSocket message for {session_id}: {e}")
                error_response = ResponseHandler.format_error_response(
                    session_id, "Error handling request", str(e)
                )
                await websocket.send_text(json.dumps(error_response))
            finally:
                db.close()
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")

async def handle_websocket_message(
    websocket: WebSocket,
    db: Session,
    session_id: str,
    incoming: str,
    context: Dict[str, Any]
) -> None:
    """Handle individual WebSocket messages."""
    payload = json.loads(incoming)
    ChatManager.save_chat(db, session_id, "user", incoming)
    
    if "company_name" in payload:
        # Handle booth request
        await handle_booth_request(websocket, db, session_id, payload, context)
    else:
        # Handle follow-up question
        await handle_followup_question(websocket, db, session_id, payload, context)

async def handle_booth_request(
    websocket: WebSocket,
    db: Session,
    session_id: str,
    payload: Dict[str, Any],
    context: Dict[str, Any]
) -> None:
    """Handle booth request in WebSocket."""
    booth_request = BoothRequest(**payload)
    response = await analyze_booth_request(booth_request)
    
    # Update context
    context["last_booth"] = booth_request
    context["last_analytics"] = response.analytics
    
    # Save session context
    ChatManager.save_session_context(db, session_id, booth_request, response.analytics)
    
    # Save and send response
    ChatManager.save_chat(db, session_id, "bot", response.model_dump_json())
    response_data = ResponseHandler.format_booth_response(response, session_id)
    await websocket.send_text(json.dumps(response_data))

async def handle_followup_question(
    websocket: WebSocket,
    db: Session,
    session_id: str,
    payload: Dict[str, Any],
    context: Dict[str, Any]
) -> None:
    """Handle follow-up question in WebSocket."""
    question = payload.get("message", "")
    
    if not context.get("last_booth") or not context.get("last_analytics"):
        # Fallback: extract context from chat history
        context = await extract_context_from_history(db, session_id)
    
    # Get AI response
    chat_response = await chat_with_llama(question, context=context)
    parsed_response = ResponseHandler.safe_json_parse(chat_response)
    
    # Save and send response
    ChatManager.save_chat(db, session_id, "bot", json.dumps(parsed_response))
    response_data = ResponseHandler.format_chat_response(parsed_response, session_id)
    await websocket.send_text(json.dumps(response_data))

async def extract_context_from_history(db: Session, session_id: str) -> Dict[str, Any]:
    """Extract context from chat history when session context is missing."""
    booth_request_msg = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id,
        ChatHistory.role == "user"
    ).order_by(ChatHistory.timestamp).first()
    
    booth_response_msg = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id,
        ChatHistory.role == "bot"
    ).order_by(ChatHistory.timestamp).first()
    
    context_str = "\n".join([
        f"User: {booth_request_msg.message}" if booth_request_msg else "",
        f"Bot: {booth_response_msg.message}" if booth_response_msg else ""
    ])
    
    return {"context": Config.SYSTEM_PROMPT + "\n" + context_str}

# Chat History Endpoints
@app.get("/chat_history")
def get_all_chat_history(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get all chat messages."""
    records = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "role": r.role,
            "message": r.message,
            "timestamp": r.timestamp.isoformat()
        }
        for r in records
    ]

@app.get("/chat_history/{session_id}")
def get_chat_history(session_id: str, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get chat messages for a specific session."""
    history = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.timestamp).all()
    
    return [
        {
            "id": h.id,
            "role": h.role,
            "message": h.message,
            "timestamp": h.timestamp.isoformat()
        }
        for h in history
    ]

# Analytics Endpoints
@app.get("/api/analytics/competitors")
def get_competitor_analytics() -> Dict[str, Any]:
    """Get competitor analytics data."""
    data = AnalyticsManager.get_json_data("enhanced_competitor_data_saudi.json")
    return AnalyticsManager.analyze_competitor_data(data)

@app.get("/api/analytics/vendor-outcomes")
def get_vendor_outcomes_analytics() -> Dict[str, Any]:
    """Get vendor outcomes analytics data."""
    data = AnalyticsManager.get_json_data("vendor_event_outcomes.json")
    return AnalyticsManager.analyze_vendor_outcomes(data)

@app.get("/api/analytics/audience-profiles")
def get_audience_profiles_analytics() -> Dict[str, Any]:
    """Get audience profiles analytics data."""
    data = AnalyticsManager.get_json_data("audience_profiles.json")
    if not data:
        return {"data": [], "analytics": {}, "ai_insights": []}
    
    best_match = max(data, key=lambda x: x.get("match_score", 0))
    low_match = min(data, key=lambda x: x.get("match_score", 1))
    
    ai_insights = [
        f"Best audience match: {best_match['event_name']} ({best_match['match_score']*100:.1f}%)",
        f"Lowest audience match: {low_match['event_name']} ({low_match['match_score']*100:.1f}%)"
    ]
    
    return {"data": data, "analytics": {}, "ai_insights": ai_insights}

@app.get("/api/analytics/global-trends")
def get_global_trends_analytics() -> Dict[str, Any]:
    """Get global trends analytics data."""
    data = AnalyticsManager.get_json_data("global_vendor_trends.json")
    if not data:
        return {"data": [], "analytics": {}, "ai_insights": []}
    
    top_growth = max(data, key=lambda x: x.get("growth_score", 0))
    ai_insights = [
        f"Fastest growing industry: {top_growth['industry']} (score {top_growth['growth_score']})"
    ]
    
    return {"data": data, "analytics": {}, "ai_insights": ai_insights}

@app.get("/api/analytics/event-schedule")
def get_event_schedule_analytics() -> Dict[str, Any]:
    """Get event schedule analytics data."""
    data = AnalyticsManager.get_json_data("enhanced_event_schedule_saudi.json")
    if not data:
        return {"data": [], "analytics": {}, "ai_insights": []}
    
    top_footfall = max(data, key=lambda x: x.get("expected_footfall", 0))
    top_international = max(data, key=lambda x: x.get("international_attendees_pct", 0))
    avg_cost = sum(x.get("cost_per_sqm", 0) for x in data) / len(data)
    tier1_events = [x for x in data if x.get("tier") == "Tier 1"]
    
    ai_insights = [
        f"Highest footfall: {top_footfall['event_name']} ({top_footfall['expected_footfall']:,} attendees)",
        f"Most international: {top_international['event_name']} ({top_international['international_attendees_pct']:.0%} international)",
        f"Average cost per sqm: {avg_cost:.0f} SAR",
        f"{len(tier1_events)} Tier 1 events with premium positioning opportunities"
    ]
    
    return {"data": data, "analytics": {"avg_cost_per_sqm": avg_cost, "tier1_count": len(tier1_events)}, "ai_insights": ai_insights}

@app.get("/api/analytics/resource-calendar")
def get_resource_calendar() -> Dict[str, Any]:
    """Get resource calendar analytics data."""
    data = AnalyticsManager.get_json_data("resource_calendar.json")
    if not data:
        return {"data": [], "analytics": {}, "ai_insights": []}
    
    # Calculate analytics
    avg_crews = sum(x["available_crews"] for x in data) / len(data)
    city_stats = {}
    for x in data:
        city = x["city"]
        city_stats.setdefault(city, []).append(x["available_crews"])
    
    city_avgs = {city: sum(vals)/len(vals) for city, vals in city_stats.items()}
    highest_city = max(city_avgs, key=city_avgs.get)
    lowest_city = min(city_avgs, key=city_avgs.get)
    zero_crew_days = [x for x in data if x["available_crews"] == 0]
    
    ai_insights = [
        f"Average available crews: {avg_crews:.2f}",
        f"City with highest avg crews: {highest_city} ({city_avgs[highest_city]:.2f})",
        f"City with lowest avg crews: {lowest_city} ({city_avgs[lowest_city]:.2f})",
        f"{len(zero_crew_days)} days had zero available crews."
    ]
    
    analytics = {"avg_crews": avg_crews, "city_avgs": city_avgs}
    
    return {"data": data, "analytics": analytics, "ai_insights": ai_insights}

# Health check endpoint
@app.get("/api/analytics/market-intelligence")
def get_market_intelligence() -> Dict[str, Any]:
    """Get Saudi market intelligence data."""
    data = AnalyticsManager.get_json_data("saudi_market_intelligence.json")
    if not data:
        return {"data": {}, "ai_insights": []}
    
    economic = data.get("economic_indicators", {})
    vision_progress = data.get("vision_2030_priorities", {})
    
    ai_insights = [
        f"GDP Growth Rate: {economic.get('gdp_growth_rate', 0)}% - Strong economic momentum",
        f"Non-oil GDP Growth: {economic.get('non_oil_gdp_growth', 0)}% - Diversification success",
        f"Foreign Investment: ${economic.get('foreign_investment_inflow_usd', 0)/1e9:.1f}B - High investor confidence",
        f"Vision 2030 Progress: {economic.get('vision_2030_progress', 0):.0%} - Transformation on track"
    ]
    
    return {"data": data, "ai_insights": ai_insights}

@app.get("/api/analytics/regulatory-environment")  
def get_regulatory_environment() -> Dict[str, Any]:
    """Get Saudi regulatory environment data."""
    data = AnalyticsManager.get_json_data("saudi_business_regulations.json")
    if not data:
        return {"data": {}, "ai_insights": []}
    
    investment_reqs = data.get("investment_requirements", {})
    incentives = data.get("incentive_programs", {})
    
    sectors_with_partnerships = sum(1 for sector, required in investment_reqs.get("local_partner_required", {}).items() if required)
    vision_sectors = len(incentives.get("vision_2030_sectors", {}).get("eligible_industries", []))
    
    ai_insights = [
        f"{sectors_with_partnerships} sectors require local partnerships",
        f"{vision_sectors} industries eligible for Vision 2030 incentives",
        "NEOM special zone offers 100% foreign ownership",
        "SME support programs provide up to 80% loan guarantees"
    ]
    
    return {"data": data, "analytics": {"partnership_sectors": sectors_with_partnerships, "vision_sectors": vision_sectors}, "ai_insights": ai_insights}

# Booth Design Endpoints
@app.post("/api/booth-design/save")
async def save_booth_design(
    request: BoothDesignRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Save a booth design to the database."""
    try:
        # Check if design already exists for this session
        existing_design = db.query(BoothDesign).filter(
            BoothDesign.session_id == request.session_id,
            BoothDesign.is_active == 1
        ).first()
        
        if existing_design:
            # Update existing design
            existing_design.design_name = request.design_name
            existing_design.design_data = request.design_data.dict()
            existing_design.updated_at = datetime.utcnow()
        else:
            # Create new design
            new_design = BoothDesign(
                session_id=request.session_id,
                design_name=request.design_name,
                design_data=request.design_data.dict()
            )
            db.add(new_design)
        
        db.commit()
        logger.info(f"Saved booth design for session {request.session_id}")
        
        return {
            "status": "success",
            "message": "Booth design saved successfully",
            "session_id": request.session_id
        }
        
    except Exception as e:
        logger.error(f"Error saving booth design: {e}")
        db.rollback()
        return {
            "status": "error",
            "message": "Failed to save booth design",
            "error": str(e)
        }

@app.get("/api/booth-design/{session_id}")
async def get_booth_design(
    session_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get booth design for a specific session."""
    try:
        design = db.query(BoothDesign).filter(
            BoothDesign.session_id == session_id,
            BoothDesign.is_active == 1
        ).first()
        
        if design:
            return {
                "status": "success",
                "design": {
                    "id": design.id,
                    "session_id": design.session_id,
                    "design_name": design.design_name,
                    "design_data": design.design_data,
                    "created_at": design.created_at.isoformat(),
                    "updated_at": design.updated_at.isoformat()
                }
            }
        else:
            return {
                "status": "not_found",
                "message": "No design found for this session"
            }
            
    except Exception as e:
        logger.error(f"Error retrieving booth design: {e}")
        return {
            "status": "error",
            "message": "Failed to retrieve booth design",
            "error": str(e)
        }

@app.post("/api/booth-design/chat")
async def design_chat(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Handle design assistant chat messages."""
    try:
        session_id = request.get("session_id")
        message = request.get("message")
        current_design = request.get("current_design", {})
        
        if not session_id or not message:
            return {
                "status": "error",
                "message": "Missing session_id or message"
            }
        
        # Get AI response for design questions
        ai_response = await chat_with_design_assistant(message, current_design, session_id)
        
        # Save chat message
        ChatManager.save_chat(db, session_id, "user", message)
        ChatManager.save_chat(db, session_id, "designer", ai_response)
        
        return {
            "status": "success",
            "response": ai_response,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error in design chat: {e}")
        return {
            "status": "error",
            "message": "Failed to process design chat",
            "error": str(e)
        }

# Approval Decision Endpoints
@app.post("/api/approval/save")
async def save_approval_decision(
    approval_data: dict,
    db: Session = Depends(get_db)
):
    """Save an approval decision."""
    try:
        session_id = approval_data.get("session_id")
        decision = approval_data.get("decision")
        decision_by = approval_data.get("decision_by", "Admin")
        decision_reason = approval_data.get("decision_reason", "")
        ai_recommendation = approval_data.get("ai_recommendation")
        ai_confidence = approval_data.get("ai_confidence")
        
        if not session_id or not decision:
            raise HTTPException(status_code=400, detail="session_id and decision are required")
        
        # Check if approval decision already exists
        existing_decision = db.query(ApprovalDecision).filter_by(session_id=session_id).first()
        
        if existing_decision:
            # Update existing decision
            existing_decision.decision = decision
            existing_decision.decision_by = decision_by
            existing_decision.decision_reason = decision_reason
            existing_decision.updated_at = datetime.utcnow()
        else:
            # Create new decision
            new_decision = ApprovalDecision(
                session_id=session_id,
                decision=decision,
                decision_by=decision_by,
                decision_reason=decision_reason,
                ai_recommendation=ai_recommendation,
                ai_confidence=json.dumps(ai_confidence) if ai_confidence else None
            )
            db.add(new_decision)
        
        db.commit()
        logger.info(f"Saved approval decision for session {session_id}: {decision}")
        
        return {
            "status": "success",
            "message": "Approval decision saved successfully",
            "session_id": session_id,
            "decision": decision
        }
        
    except Exception as e:
        logger.error(f"Error saving approval decision: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save approval decision: {str(e)}")

@app.get("/api/approval/{session_id}")
async def get_approval_decision(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get approval decision for a specific session."""
    try:
        decision = db.query(ApprovalDecision).filter_by(session_id=session_id).first()
        
        if decision:
            return {
                "status": "success",
                "decision": {
                    "session_id": decision.session_id,
                    "decision": decision.decision,
                    "decision_by": decision.decision_by,
                    "decision_reason": decision.decision_reason,
                    "ai_recommendation": decision.ai_recommendation,
                    "ai_confidence": json.loads(decision.ai_confidence) if decision.ai_confidence else None,
                    "created_at": decision.created_at.isoformat(),
                    "updated_at": decision.updated_at.isoformat()
                }
            }
        else:
            return {
                "status": "not_found",
                "message": "No approval decision found for this session"
            }
            
    except Exception as e:
        logger.error(f"Error retrieving approval decision: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve approval decision: {str(e)}")

@app.get("/api/approvals/all")
async def get_all_approval_decisions(
    db: Session = Depends(get_db)
):
    """Get all approval decisions."""
    try:
        decisions = db.query(ApprovalDecision).order_by(ApprovalDecision.updated_at.desc()).all()
        
        return {
            "status": "success",
            "decisions": [
                {
                    "session_id": d.session_id,
                    "decision": d.decision,
                    "decision_by": d.decision_by,
                    "decision_reason": d.decision_reason,
                    "ai_recommendation": d.ai_recommendation,
                    "created_at": d.created_at.isoformat(),
                    "updated_at": d.updated_at.isoformat()
                }
                for d in decisions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving approval decisions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve approval decisions: {str(e)}")

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)