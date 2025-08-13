from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Enum, JSON, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

DATABASE_URL = "mysql+pymysql://root:password@127.0.0.1:3306/ai_booth_agent"

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    role = Column(Enum("user", "bot", "designer"), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class SessionContext(Base):
    __tablename__ = "session_context"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    last_booth = Column(Text, nullable=True)  # Store as JSON string
    last_analytics = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class BoothDesign(Base):
    __tablename__ = "booth_designs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    design_name = Column(String(255), nullable=False)
    design_data = Column(JSON, nullable=False)  # Store complete design as JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = Column(Integer, default=1)  # For soft delete

class ApprovalDecision(Base):
    __tablename__ = "approval_decisions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    decision = Column(Enum("approved", "rejected", "needs_review", "pending"), nullable=False, default="pending")
    decision_by = Column(String(255), nullable=True)  # Who made the decision
    decision_reason = Column(Text, nullable=True)  # Optional reason/notes
    ai_recommendation = Column(String(50), nullable=True)  # Original AI recommendation
    ai_confidence = Column(Text, nullable=True)  # Store AI analytics as JSON string
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)