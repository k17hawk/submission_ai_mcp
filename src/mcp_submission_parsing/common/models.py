"""Shared data models for all MCP components"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ClaimStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    UNDER_REVIEW = "UNDER_REVIEW"
    WITH_SIU = "WITH_SIU"
    COMPLETED = "COMPLETED"


@dataclass
class ParsedClaim:
    """Output from Agent 1 - Submission Parser"""
    policy_number: Optional[str] = None
    incident_date: Optional[str] = None
    incident_type: Optional[str] = None
    incident_severity: Optional[str] = None
    incident_location: Optional[str] = None
    incident_city: Optional[str] = None
    incident_state: Optional[str] = None
    auto_make: Optional[str] = None
    auto_model: Optional[str] = None
    auto_year: Optional[int] = None
    authorities_contacted: Optional[str] = None
    collision_type: Optional[str] = None
    number_of_vehicles_involved: Optional[int] = None
    bodily_injuries: Optional[int] = None
    witnesses: Optional[int] = None
    police_report_available: Optional[str] = None
    property_damage: Optional[str] = None
    total_claim_amount: Optional[float] = None
    incident_hour_of_the_day: Optional[int] = None
    vehicle_claim: Optional[float] = None
    injury_claim: Optional[float] = None
    property_claim: Optional[float] = None
    
    extraction_confidence: float = 0.0
    missing_fields: List[str] = field(default_factory=list)
    llm_enhanced: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyVerification:
    """Output from Agent 2 - Policy Lookup"""
    found: bool
    policy_number: str
    is_active: bool = False
    policy_status: str = "Unknown"
    incident_in_policy_period: bool = False
    effective_date: str = ""
    expiration_date: str = ""
    customer_id: str = ""
    customer_name: str = ""
    coverage_code: str = ""
    coverage_name: str = ""
    coverage_limit: float = 0.0
    deductible: float = 0.0
    vehicle_make: str = ""
    vehicle_model: str = ""
    vehicle_year: int = 0
    vehicle_value: float = 0.0
    prior_claims: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    """Output from Agent 3 - Risk Rules"""
    passed: bool
    risk_score: float
    risk_level: str
    violations: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    requires_siu: bool = False
    requires_adjuster: bool = False
    auto_decision: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureVector:
    """Output from Agent 4 - Feature Builder"""
    features: Dict[str, Any]
    feature_count: int
    imputed_count: int
    imputed_fields: List[str]
    ready_for_ml: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FraudPrediction:
    """Output from Agent 5 - Fraud Detection"""
    fraud_probability: float
    fraud_flag: str
    risk_level: str
    threshold_used: float
    requires_siu: bool
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompleteClaimResponse:
    """Final output from orchestrator"""
    request_id: str
    timestamp: str
    status: ClaimStatus
    parsed_claim: ParsedClaim
    policy_verification: PolicyVerification
    risk_assessment: RiskAssessment
    fraud_prediction: FraudPrediction
    final_decision: str
    recommended_action: str
    summary: str
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'parsed_claim': self.parsed_claim.to_dict(),
            'policy_verification': self.policy_verification.to_dict(),
            'risk_assessment': self.risk_assessment.to_dict(),
            'fraud_prediction': self.fraud_prediction.to_dict(),
            'final_decision': self.final_decision,
            'recommended_action': self.recommended_action,
            'summary': self.summary,
            'processing_time_ms': self.processing_time_ms
        }