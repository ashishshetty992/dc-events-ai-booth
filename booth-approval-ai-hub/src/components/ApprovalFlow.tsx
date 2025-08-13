import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  TrendingUp, 
  DollarSign, 
  Users, 
  MapPin, 
  Calendar,
  Building,
  BarChart3,
  Shield,
  Sparkles,
  ArrowRight,
  ThumbsUp,
  ThumbsDown,
  Clock,
  FileText,
  Target,
  Globe,
  Flag
} from 'lucide-react';
import { API_ENDPOINTS } from '@/lib/api';
import Sidebar from './Sidebar';

interface BoothRequest {
  company_name: string;
  booth_size_sqm: number;
  budget_inr: number;
  event_name: string;
  event_date: string;
  industry: string;
  location: string;
  past_events?: string[];
  product_description?: string;
  target_audience?: string;
  objectives?: string;
}

interface ApprovalAnalytics {
  ml_approval_probability: number;
  ml_predicted_roi: number;
  market_opportunity_score: number;
  competition_intensity: string;
  regulatory_complexity: string;
  vision_2030_alignment: string[];
  strategic_recommendations: string[];
  risk_factors: string[];
  market_opportunities: string[];
  confidence_score: number;
}

interface ApprovalResponse {
  summary: string;
  approval_recommendation: 'Approve' | 'Reject' | 'Needs Review';
  competitor_analysis: string;
  risk_flags: string[];
  analytics: ApprovalAnalytics;
}

interface ApprovalDecision {
  session_id: string;
  decision: 'approved' | 'rejected' | 'needs_review' | 'pending';
  decision_by: string;
  decision_reason: string;
  ai_recommendation: string;
  created_at: string;
  updated_at: string;
}

const ApprovalFlow: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [approvalData, setApprovalData] = useState<ApprovalResponse | null>(null);
  const [boothRequest, setBoothRequest] = useState<BoothRequest | null>(null);
  const [currentDecision, setCurrentDecision] = useState<ApprovalDecision | null>(null);
  const [loading, setLoading] = useState(true);
  const [approvalAction, setApprovalAction] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    if (sessionId) {
      fetchApprovalData();
    }
  }, [sessionId]);

  const fetchApprovalData = async () => {
    try {
      // Get chat history to extract booth request and AI response
      const historyResponse = await fetch(API_ENDPOINTS.CHAT_HISTORY_BY_ID(sessionId));
      const history = await historyResponse.json();
      
      // Get existing approval decision if any
      try {
        const decisionResponse = await fetch(API_ENDPOINTS.APPROVAL_BY_ID(sessionId));
        const decisionData = await decisionResponse.json();
        if (decisionData.status === 'success') {
          setCurrentDecision(decisionData.decision);
        }
      } catch (decisionError) {
        console.log('No existing approval decision found');
      }
      
      if (history && history.length >= 2) {
        // Find the booth request (user message) and AI response
        const userMessage = history.find((msg: any) => msg.role === 'user');
        const botMessage = history.find((msg: any) => msg.role === 'bot');
        
        if (userMessage && botMessage) {
          try {
            const requestData = JSON.parse(userMessage.message);
            const responseData = JSON.parse(botMessage.message);
            
            setBoothRequest(requestData);
            setApprovalData(responseData);
          } catch (parseError) {
            console.error('Error parsing data:', parseError);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching approval data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleApprovalAction = async (action: 'approve' | 'reject' | 'request_review') => {
    setActionLoading(true);
    
    try {
      const decisionData = {
        session_id: sessionId,
        decision: action === 'approve' ? 'approved' : action === 'reject' ? 'rejected' : 'needs_review',
        decision_by: 'Admin', // In a real app, this would be the current user
        decision_reason: `Decision made via approval flow: ${action}`,
        ai_recommendation: approvalData?.approval_recommendation,
        ai_confidence: approvalData?.analytics
      };

      const response = await fetch(API_ENDPOINTS.APPROVAL_SAVE, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(decisionData),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Approval decision saved:', result);
        setApprovalAction(action);
        
        // Refresh the current decision
        const decisionResponse = await fetch(API_ENDPOINTS.APPROVAL_BY_ID(sessionId));
        const updatedDecision = await decisionResponse.json();
        if (updatedDecision.status === 'success') {
          setCurrentDecision(updatedDecision.decision);
        }
        
        setTimeout(() => {
          navigate('/dashboard');
        }, 2000);
      } else {
        throw new Error('Failed to save approval decision');
      }
    } catch (error) {
      console.error('Error saving approval decision:', error);
      alert('Failed to save approval decision. Please try again.');
    } finally {
      setActionLoading(false);
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'Approve': return 'text-green-600 bg-green-50 border-green-200';
      case 'Reject': return 'text-red-600 bg-red-50 border-red-200';
      case 'Needs Review': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation) {
      case 'Approve': return <CheckCircle className="w-6 h-6" />;
      case 'Reject': return <XCircle className="w-6 h-6" />;
      case 'Needs Review': return <AlertTriangle className="w-6 h-6" />;
      default: return <Clock className="w-6 h-6" />;
    }
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'approved': return 'text-green-600 bg-green-50 border-green-200';
      case 'rejected': return 'text-red-600 bg-red-50 border-red-200';
      case 'needs_review': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'pending': return 'text-gray-600 bg-gray-50 border-gray-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getDecisionIcon = (decision: string) => {
    switch (decision) {
      case 'approved': return <CheckCircle className="w-5 h-5" />;
      case 'rejected': return <XCircle className="w-5 h-5" />;
      case 'needs_review': return <AlertTriangle className="w-5 h-5" />;
      case 'pending': return <Clock className="w-5 h-5" />;
      default: return <Clock className="w-5 h-5" />;
    }
  };

  const getDecisionLabel = (decision: string) => {
    switch (decision) {
      case 'approved': return 'Approved';
      case 'rejected': return 'Rejected';
      case 'needs_review': return 'Needs Review';
      case 'pending': return 'Pending';
      default: return 'Unknown';
    }
  };

  if (loading) {
    return (
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
        <Sidebar />
        <div className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading approval data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!approvalData || !boothRequest) {
    return (
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
        <Sidebar />
        <div className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">No Approval Data Found</h2>
            <p className="text-gray-600 mb-6">Unable to load approval data for this session.</p>
            <button 
              onClick={() => navigate('/dashboard')}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (approvalAction) {
    return (
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
        <Sidebar />
        <div className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Action Processed Successfully
            </h2>
            <p className="text-gray-600 mb-6">
              Your approval decision has been recorded and the applicant will be notified.
            </p>
            <div className="animate-pulse">Redirecting to dashboard...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
      <Sidebar />
      
      <div className="flex-1 ml-64 overflow-y-auto">
        {/* Header */}
        <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 px-8 py-6 shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-800 via-blue-600 to-purple-600 bg-clip-text text-transparent">
                Booth Approval Review
              </h1>
              <p className="text-gray-600 mt-1">AI-Powered Analysis & Recommendation</p>
            </div>
            <div className="flex items-center space-x-3">
              <span className="text-sm text-gray-500">Session: {sessionId}</span>
              {currentDecision ? (
                <div className={`px-4 py-2 rounded-full border-2 font-semibold flex items-center space-x-2 ${getDecisionColor(currentDecision.decision)}`}>
                  {getDecisionIcon(currentDecision.decision)}
                  <span>{getDecisionLabel(currentDecision.decision)}</span>
                  <span className="text-xs opacity-75">({currentDecision.decision_by})</span>
                </div>
              ) : (
                <div className={`px-4 py-2 rounded-full border-2 font-semibold flex items-center space-x-2 ${getRecommendationColor(approvalData.approval_recommendation)}`}>
                  {getRecommendationIcon(approvalData.approval_recommendation)}
                  <span>AI Recommends: {approvalData.approval_recommendation}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="p-8">
          {/* AI Confidence & Key Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-blue-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800">AI Confidence</h3>
                <Sparkles className="w-5 h-5 text-purple-500" />
              </div>
              <div className="text-3xl font-bold text-purple-600 mb-2">
                {Math.round((approvalData.analytics?.confidence_score || 0) * 100)}%
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-1000"
                  style={{ width: `${(approvalData.analytics?.confidence_score || 0) * 100}%` }}
                />
              </div>
            </div>

            <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-green-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800">Approval Probability</h3>
                <TrendingUp className="w-5 h-5 text-green-500" />
              </div>
              <div className="text-3xl font-bold text-green-600 mb-2">
                {Math.round((approvalData.analytics?.ml_approval_probability || 0) * 100)}%
              </div>
              <div className="text-sm text-gray-600">ML Prediction</div>
            </div>

            <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-blue-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800">Expected ROI</h3>
                <DollarSign className="w-5 h-5 text-blue-500" />
              </div>
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {approvalData.analytics?.ml_predicted_roi?.toFixed(1) || 0}x
              </div>
              <div className="text-sm text-gray-600">Predicted Return</div>
            </div>

            <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-orange-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800">Market Score</h3>
                <Target className="w-5 h-5 text-orange-500" />
              </div>
              <div className="text-3xl font-bold text-orange-600 mb-2">
                {Math.round((approvalData.analytics?.market_opportunity_score || 0) * 100)}%
              </div>
              <div className="text-sm text-gray-600">Opportunity Rating</div>
            </div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column - Request Details */}
            <div className="lg:col-span-1 space-y-6">
              {/* Company Information */}
              <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="flex items-center mb-4">
                  <Building className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Company Details</h3>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm text-gray-500">Company Name</label>
                    <p className="font-medium text-gray-800">{boothRequest.company_name}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-500">Industry</label>
                    <p className="font-medium text-gray-800">{boothRequest.industry}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-500">Event</label>
                    <p className="font-medium text-gray-800">{boothRequest.event_name}</p>
                  </div>
                  <div className="flex items-center text-sm text-gray-600">
                    <Calendar className="w-4 h-4 mr-1" />
                    <span>{boothRequest.event_date}</span>
                    <MapPin className="w-4 h-4 ml-3 mr-1" />
                    <span>{boothRequest.location}</span>
                  </div>
                </div>
              </div>

              {/* Booth Specifications */}
              <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="flex items-center mb-4">
                  <BarChart3 className="w-5 h-5 mr-2 text-purple-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Booth Specs</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Size</span>
                    <span className="font-semibold">{boothRequest.booth_size_sqm} sqm</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Budget</span>
                    <span className="font-semibold">SAR {boothRequest.budget_inr?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cost per sqm</span>
                    <span className="font-semibold">SAR {Math.round((boothRequest.budget_inr || 0) / boothRequest.booth_size_sqm).toLocaleString()}</span>
                  </div>
                </div>
              </div>

              {/* Risk Assessment */}
              <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="flex items-center mb-4">
                  <Shield className="w-5 h-5 mr-2 text-red-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Risk Assessment</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Competition</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      approvalData.analytics?.competition_intensity === 'Low' ? 'bg-green-100 text-green-700' :
                      approvalData.analytics?.competition_intensity === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {approvalData.analytics?.competition_intensity || 'Unknown'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Regulatory</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      approvalData.analytics?.regulatory_complexity === 'Low' ? 'bg-green-100 text-green-700' :
                      approvalData.analytics?.regulatory_complexity === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {approvalData.analytics?.regulatory_complexity || 'Unknown'}
                    </span>
                  </div>
                </div>
                {approvalData.risk_flags && approvalData.risk_flags.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Flags:</h4>
                    <div className="space-y-1">
                      {approvalData.risk_flags.map((risk, index) => (
                        <div key={index} className="flex items-center text-sm text-red-600">
                          <Flag className="w-3 h-3 mr-1" />
                          <span>{risk}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - AI Analysis */}
            <div className="lg:col-span-2 space-y-6">
              {/* AI Summary */}
              <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="flex items-center mb-4">
                  <Sparkles className="w-5 h-5 mr-2 text-purple-600" />
                  <h3 className="text-lg font-semibold text-gray-800">AI Analysis Summary</h3>
                </div>
                <div className="prose prose-sm max-w-none">
                  <p className="text-gray-700 leading-relaxed">{approvalData.summary}</p>
                </div>
              </div>

              {/* Competitor Analysis */}
              <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="flex items-center mb-4">
                  <Globe className="w-5 h-5 mr-2 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-800">Market & Competition Analysis</h3>
                </div>
                <div className="prose prose-sm max-w-none">
                  <p className="text-gray-700 leading-relaxed">{approvalData.competitor_analysis}</p>
                </div>
              </div>

              {/* Strategic Recommendations */}
              {approvalData.analytics?.strategic_recommendations && approvalData.analytics.strategic_recommendations.length > 0 && (
                <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
                  <div className="flex items-center mb-4">
                    <Target className="w-5 h-5 mr-2 text-green-600" />
                    <h3 className="text-lg font-semibold text-gray-800">Strategic Recommendations</h3>
                  </div>
                  <div className="space-y-2">
                    {approvalData.analytics.strategic_recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start">
                        <ArrowRight className="w-4 h-4 text-green-500 mt-1 mr-2 flex-shrink-0" />
                        <span className="text-gray-700">{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Vision 2030 Alignment */}
              {approvalData.analytics?.vision_2030_alignment && approvalData.analytics.vision_2030_alignment.length > 0 && (
                <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-purple-100">
                  <div className="flex items-center mb-4">
                    <Flag className="w-5 h-5 mr-2 text-purple-600" />
                    <h3 className="text-lg font-semibold text-gray-800">Vision 2030 Alignment</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {approvalData.analytics.vision_2030_alignment.map((alignment, index) => (
                      <div key={index} className="flex items-center p-3 bg-purple-50 rounded-lg">
                        <CheckCircle className="w-4 h-4 text-purple-600 mr-2" />
                        <span className="text-sm text-purple-800">{alignment}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Current Decision Status */}
          {currentDecision && (
            <div className="mt-8 bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Current Decision Status</h3>
              <div className={`p-4 rounded-xl border-2 ${getDecisionColor(currentDecision.decision)}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getDecisionIcon(currentDecision.decision)}
                    <div>
                      <span className="font-semibold text-lg">{getDecisionLabel(currentDecision.decision)}</span>
                      <p className="text-sm opacity-75">Decision by: {currentDecision.decision_by}</p>
                      <p className="text-sm opacity-75">Date: {new Date(currentDecision.updated_at).toLocaleString()}</p>
                    </div>
                  </div>
                  {currentDecision.decision_reason && (
                    <div className="text-right max-w-md">
                      <p className="text-sm font-medium">Reason:</p>
                      <p className="text-sm opacity-75">{currentDecision.decision_reason}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-8 bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              {currentDecision ? 'Update Decision' : 'Approval Actions'}
            </h3>
            <div className="flex flex-wrap gap-4">
              <button
                onClick={() => handleApprovalAction('approve')}
                disabled={actionLoading}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl hover:from-green-600 hover:to-green-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {actionLoading ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                ) : (
                  <ThumbsUp className="w-5 h-5 mr-2" />
                )}
                Approve Request
              </button>
              
              <button
                onClick={() => handleApprovalAction('reject')}
                disabled={actionLoading}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl hover:from-red-600 hover:to-red-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {actionLoading ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                ) : (
                  <ThumbsDown className="w-5 h-5 mr-2" />
                )}
                Reject Request
              </button>
              
              <button
                onClick={() => handleApprovalAction('request_review')}
                disabled={actionLoading}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-yellow-500 to-yellow-600 text-white rounded-xl hover:from-yellow-600 hover:to-yellow-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {actionLoading ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                ) : (
                  <FileText className="w-5 h-5 mr-2" />
                )}
                Request Additional Review
              </button>
              
              <button
                onClick={() => navigate(`/chat/${sessionId}`)}
                disabled={actionLoading}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                <Users className="w-5 h-5 mr-2" />
                Chat with Applicant
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApprovalFlow;
