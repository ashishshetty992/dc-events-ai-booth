import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Building, 
  TrendingUp, 
  Users, 
  Clock, 
  Plus, 
  Eye,
  MessageCircle,
  BarChart3,
  Sparkles,
  ArrowRight,
  Calendar,
  MapPin
} from 'lucide-react';
import Sidebar from './Sidebar';

interface DashboardStats {
  totalProjects: number;
  activeChats: number;
  monthlyBudget: number;
  completionRate: number;
}

interface RecentSession {
  id: string;
  company: string;
  event: string;
  location: string;
  date: string;
  status: 'active' | 'completed' | 'pending';
  budget: number;
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<DashboardStats>({
    totalProjects: 0,
    activeChats: 0,
    monthlyBudget: 0,
    completionRate: 0
  });
  const [recentSessions, setRecentSessions] = useState<RecentSession[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Simulate API calls - in real app these would be actual API calls
      setStats({
        totalProjects: 24,
        activeChats: 7,
        monthlyBudget: 2450000,
        completionRate: 89
      });

      setRecentSessions([
        {
          id: 'session_1755092161219',
          company: 'TechCorp Saudi',
          event: 'LEAP Technology Conference',
          location: 'Riyadh',
          date: '2024-08-15',
          status: 'active',
          budget: 125000
        },
        {
          id: 'session_1755092161220',
          company: 'HealthInnovate',
          event: 'Saudi Healthcare Exhibition',
          location: 'Jeddah',
          date: '2024-08-20',
          status: 'completed',
          budget: 89000
        },
        {
          id: 'session_1755092161221',
          company: 'GreenEnergy Co',
          event: 'Saudi Green Initiative Summit',
          location: 'NEOM',
          date: '2024-08-25',
          status: 'pending',
          budget: 156000
        }
      ]);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewProject = () => {
    navigate('/');
  };

  const handleViewSession = (sessionId: string) => {
    navigate(`/chat/${sessionId}`);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-700';
      case 'completed': return 'bg-blue-100 text-blue-700';
      case 'pending': return 'bg-yellow-100 text-yellow-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  if (loading) {
    return (
      <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
        <Sidebar />
        <div className="flex-1 ml-64 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading dashboard...</p>
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
                Saudi Booth AI Dashboard
              </h1>
              <p className="text-gray-600 mt-1">Manage your exhibition booth projects and analyze performance</p>
            </div>
            <button
              onClick={handleNewProject}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white rounded-xl hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              <Plus className="w-5 h-5" />
              <span className="font-medium">New Booth Project</span>
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="p-8">
          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-500 text-sm font-medium">Total Projects</p>
                  <p className="text-3xl font-bold text-gray-800">{stats.totalProjects}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center">
                  <Building className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                <span className="text-green-600 font-medium">+12%</span>
                <span className="text-gray-500 ml-1">from last month</span>
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-500 text-sm font-medium">Active Chats</p>
                  <p className="text-3xl font-bold text-gray-800">{stats.activeChats}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center">
                  <MessageCircle className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <Clock className="w-4 h-4 text-blue-500 mr-1" />
                <span className="text-gray-500">2 sessions started today</span>
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-500 text-sm font-medium">Monthly Budget</p>
                  <p className="text-3xl font-bold text-gray-800">SAR {(stats.monthlyBudget / 1000).toFixed(0)}K</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-yellow-500 to-orange-600 rounded-2xl flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                <span className="text-green-600 font-medium">+8%</span>
                <span className="text-gray-500 ml-1">vs target</span>
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-500 text-sm font-medium">Completion Rate</p>
                  <p className="text-3xl font-bold text-gray-800">{stats.completionRate}%</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <Users className="w-4 h-4 text-blue-500 mr-1" />
                <span className="text-gray-500">21 projects completed</span>
              </div>
            </div>
          </div>

          {/* Recent Sessions */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-800">Recent Sessions</h3>
                <button 
                  onClick={() => navigate('/logs')}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center space-x-1"
                >
                  <span>View all</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-4">
                {recentSessions.map((session) => (
                  <div 
                    key={session.id}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors cursor-pointer"
                    onClick={() => handleViewSession(session.id)}
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="font-medium text-gray-800">{session.company}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(session.status)}`}>
                          {session.status}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>{session.event}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <MapPin className="w-4 h-4" />
                          <span>{session.location}</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-gray-800">SAR {session.budget.toLocaleString()}</p>
                      <p className="text-sm text-gray-500">{session.date}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Quick Actions</h3>
              
              <div className="space-y-4">
                <button 
                  onClick={handleNewProject}
                  className="w-full p-4 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-xl hover:from-blue-100 hover:to-purple-100 transition-all duration-300 text-left group"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <Plus className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-800">Create New Booth</h4>
                      <p className="text-sm text-gray-600">Start a new booth design project</p>
                    </div>
                  </div>
                </button>

                <button 
                  onClick={() => navigate('/logs')}
                  className="w-full p-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl hover:from-green-100 hover:to-emerald-100 transition-all duration-300 text-left group"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <Eye className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-800">View Session Logs</h4>
                      <p className="text-sm text-gray-600">Review past conversations and designs</p>
                    </div>
                  </div>
                </button>

                <button 
                  onClick={() => navigate('/analytics')}
                  className="w-full p-4 bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-xl hover:from-yellow-100 hover:to-orange-100 transition-all duration-300 text-left group"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-orange-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <BarChart3 className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-800">Analytics Dashboard</h4>
                      <p className="text-sm text-gray-600">View performance metrics and insights</p>
                    </div>
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
