import React, { useState, useEffect } from 'react';
import { Calendar, Filter, Search, MessageSquare, Eye, Palette, CheckCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { API_ENDPOINTS } from '@/lib/api';
import Sidebar from './Sidebar';

interface ChatHistory {
  session_id: string;
  role: 'user' | 'bot';
  message: string;
  timestamp: string;
}

interface SessionSummary {
  session_id: string;
  last_message: string;
  last_role: string;
  last_timestamp: string;
  message_count: number;
  event_name?: string;
  company_name?: string;
  approval_status?: string;
}

const SessionLogs: React.FC = () => {
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [approvalDecisions, setApprovalDecisions] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sessionFilter, setSessionFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchChatHistory();
    fetchApprovalDecisions();
  }, []);

  const fetchChatHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch(API_ENDPOINTS.CHAT_HISTORY);
      const data = await response.json();
      setChatHistory(data);
    } catch (error) {
      console.error("Error fetching chat history:", error);
      setChatHistory([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchApprovalDecisions = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.APPROVALS_ALL);
      const data = await response.json();
      if (data.status === 'success') {
        const decisionsMap: Record<string, any> = {};
        data.decisions.forEach((decision: any) => {
          decisionsMap[decision.session_id] = decision;
        });
        setApprovalDecisions(decisionsMap);
      }
    } catch (error) {
      console.error("Error fetching approval decisions:", error);
    }
  };

  // Group messages by session_id, get last message, last timestamp, and count
  const sessionMap: Record<string, SessionSummary> = {};
  chatHistory.forEach((msg) => {
    if (!sessionMap[msg.session_id]) {
      sessionMap[msg.session_id] = {
        session_id: msg.session_id,
        last_message: msg.message,
        last_role: msg.role,
        last_timestamp: msg.timestamp,
        message_count: 1,
        event_name: undefined,
        company_name: undefined,
        approval_status: approvalDecisions[msg.session_id]?.decision || 'pending',
      };
    } else {
      // Update if this message is newer
      if (new Date(msg.timestamp) > new Date(sessionMap[msg.session_id].last_timestamp)) {
        sessionMap[msg.session_id].last_message = msg.message;
        sessionMap[msg.session_id].last_role = msg.role;
        sessionMap[msg.session_id].last_timestamp = msg.timestamp;
      }
      sessionMap[msg.session_id].message_count += 1;
    }
    // For every user message, if event_name/company_name not set, try to extract
    if (msg.role === 'user') {
      try {
        const parsed = JSON.parse(msg.message);
        if (parsed.event_name && !sessionMap[msg.session_id].event_name) sessionMap[msg.session_id].event_name = parsed.event_name;
        if (parsed.company_name && !sessionMap[msg.session_id].company_name) sessionMap[msg.session_id].company_name = parsed.company_name;
        // Update approval status
        sessionMap[msg.session_id].approval_status = approvalDecisions[msg.session_id]?.decision || 'pending';
      } catch {}
    }
  });
  let sessionSummaries = Object.values(sessionMap);

  // Apply search/filter to sessions
  sessionSummaries = sessionSummaries.filter(session => {
    const matchesSearch = session.last_message.toLowerCase().includes(searchTerm.toLowerCase()) || session.session_id.includes(searchTerm) || (session.event_name?.toLowerCase().includes(searchTerm.toLowerCase()) ?? false) || (session.company_name?.toLowerCase().includes(searchTerm.toLowerCase()) ?? false);
    const matchesSession = !sessionFilter || session.session_id.includes(sessionFilter);
    const matchesDate = !dateFilter || session.last_timestamp.startsWith(dateFilter);
    return matchesSearch && matchesSession && matchesDate;
  });

  // Sort by last activity (desc)
  sessionSummaries.sort((a, b) => new Date(b.last_timestamp).getTime() - new Date(a.last_timestamp).getTime());

  const truncateMessage = (message: string, maxLength: number = 100) =>
    message.length > maxLength ? `${message.substring(0, maxLength)}...` : message;

  const formatTimestamp = (timestamp: string) =>
    new Date(timestamp).toLocaleString();

  const getApprovalStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-700';
      case 'rejected': return 'bg-red-100 text-red-700';
      case 'needs_review': return 'bg-yellow-100 text-yellow-700';
      case 'pending': return 'bg-gray-100 text-gray-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getApprovalStatusLabel = (status: string) => {
    switch (status) {
      case 'approved': return 'Approved';
      case 'rejected': return 'Rejected';
      case 'needs_review': return 'Needs Review';
      case 'pending': return 'Pending';
      default: return 'Unknown';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto" />
          <p className="mt-4 text-gray-600">Loading chat history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
      <Sidebar />
      <div className="flex-1 ml-64 flex flex-col bg-white/80 backdrop-blur-sm">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Session Logs</h2>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <MessageSquare className="w-4 h-4" />
            <span>{sessionSummaries.length} sessions</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search sessions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <select
              value={sessionFilter}
              onChange={(e) => setSessionFilter(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none"
            >
              <option value="">All Sessions</option>
              {Object.keys(sessionMap).map(session => (
                <option key={session} value={session}>{session}</option>
              ))}
            </select>
          </div>

          <div className="relative">
            <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="date"
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Session ID</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Event Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Company Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Approval Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Message</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Activity</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Messages</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sessionSummaries.map((session) => (
              <tr
                key={session.session_id}
                className="hover:bg-gray-50 cursor-pointer"
                onClick={() => navigate(`/chat/${session.session_id}`)}
              >
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600 hover:underline">
                  {session.session_id}
                </td>
                <td className="px-6 py-4 text-sm text-gray-900 max-w-xs">
                  {session.event_name || <span className="text-gray-400 italic">—</span>}
                </td>
                <td className="px-6 py-4 text-sm text-gray-900 max-w-xs">
                  {session.company_name || <span className="text-gray-400 italic">—</span>}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getApprovalStatusColor(session.approval_status || 'pending')}`}>
                    {getApprovalStatusLabel(session.approval_status || 'pending')}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-900 max-w-md">
                  <div title={session.last_message}>{truncateMessage(session.last_message)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatTimestamp(session.last_timestamp)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 text-center">
                  {session.message_count}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-center">
                  <div className="flex space-x-1">
                    <button
                      className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition text-xs"
                      onClick={e => { e.stopPropagation(); navigate(`/chat/${session.session_id}`); }}
                      title="Open Chat"
                    >
                      <Eye className="w-3 h-3 mr-1" /> Chat
                    </button>
                    <button
                      className="inline-flex items-center px-2 py-1 bg-purple-100 text-purple-700 rounded hover:bg-purple-200 transition text-xs"
                      onClick={e => { e.stopPropagation(); navigate(`/chat/${session.session_id}`); }}
                      title="Design Booth"
                    >
                      <Palette className="w-3 h-3 mr-1" /> Design
                    </button>
                    <button
                      className="inline-flex items-center px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200 transition text-xs"
                      onClick={e => { e.stopPropagation(); navigate(`/approval/${session.session_id}`); }}
                      title="Review Approval"
                    >
                      <CheckCircle className="w-3 h-3 mr-1" /> Approve
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {sessionSummaries.length === 0 && (
          <div className="text-center py-12">
            <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No sessions found</h3>
            <p className="text-gray-600">Try adjusting your search filters or start a new conversation.</p>
          </div>
        )}
      </div>
      </div>
    </div>
  );
};

export default SessionLogs;