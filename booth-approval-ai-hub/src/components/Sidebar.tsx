import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Home, 
  Layers, 
  MessageCircle, 
  Settings, 
  BarChart3, 
  FileText, 
  Users, 
  ChevronLeft, 
  ChevronRight,
  Building,
  Sparkles,
  CheckCircle
} from 'lucide-react';

interface NavItem {
  icon: React.ReactNode;
  label: string;
  path: string;
  badge?: string;
}

const Sidebar: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const navItems: NavItem[] = [
    { icon: <Home className="w-5 h-5" />, label: 'Dashboard', path: '/dashboard' },
    { icon: <Building className="w-5 h-5" />, label: 'Submit Request', path: '/' },
    { icon: <CheckCircle className="w-5 h-5" />, label: 'Approval Review', path: '/approval/latest', badge: 'AI' },
    { icon: <Layers className="w-5 h-5" />, label: 'Design Studio', path: '/chat/new', badge: 'New' },
    { icon: <FileText className="w-5 h-5" />, label: 'Session Logs', path: '/logs' },
    { icon: <BarChart3 className="w-5 h-5" />, label: 'Analytics', path: '/analytics' },
    { icon: <Users className="w-5 h-5" />, label: 'Competitors', path: '/competitors' },
    { icon: <Settings className="w-5 h-5" />, label: 'Settings', path: '/settings' },
  ];

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const handleNavigation = (path: string) => {
    if (path === '/chat/new') {
      // Generate new session for design studio
      const sessionId = `session_${Date.now()}`;
      navigate(`/chat/${sessionId}`);
    } else if (path === '/approval/latest') {
      // For now, let's redirect to the logs where users can select a session to approve
      // In a real app, you'd fetch the latest session with approval data
      navigate('/logs');
    } else {
      navigate(path);
    }
  };

  return (
    <div className={`fixed left-0 top-0 h-full bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 border-r border-slate-700 transition-all duration-300 z-50 ${
      isCollapsed ? 'w-16' : 'w-64'
    }`}>
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-white font-bold text-lg">Saudi Booth AI</h1>
                <p className="text-slate-400 text-xs">Exhibition Designer</p>
              </div>
            </div>
          )}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {navItems.map((item) => (
            <button
              key={item.path}
              onClick={() => handleNavigation(item.path)}
              className={`w-full flex items-center space-x-3 px-3 py-3 rounded-xl transition-all duration-200 group ${
                isActive(item.path)
                  ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 text-white border border-blue-500/30'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <div className={`transition-colors ${isActive(item.path) ? 'text-blue-400' : 'text-slate-400 group-hover:text-white'}`}>
                {item.icon}
              </div>
              {!isCollapsed && (
                <>
                  <span className="font-medium">{item.label}</span>
                  {item.badge && (
                    <span className="ml-auto px-2 py-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xs rounded-full">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700">
        {!isCollapsed && (
          <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-3 border border-blue-500/20">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-bold">AI</span>
              </div>
              <div>
                <p className="text-white text-sm font-medium">Saudi AI Assistant</p>
                <p className="text-slate-400 text-xs">Online & Ready</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;