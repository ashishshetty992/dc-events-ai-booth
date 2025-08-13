import React, { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';

const MainLayout = () => {
  const location = useLocation();
  const navigate = useNavigate();
  // Determine active tab from route - only for routes within MainLayout
  let activeTab = 'form'; // Default to form as home page
  if (location.pathname.startsWith('/logs')) activeTab = 'logs';
  else if (location.pathname.startsWith('/form')) activeTab = 'form';
  else if (location.pathname.startsWith('/analytics')) activeTab = 'analytics';
  else if (location.pathname === '/') activeTab = 'form'; // Home page defaults to form

  const setActiveTab = (tab: string) => {
    // Only handle routes that stay within MainLayout
    if (tab === 'logs') navigate('/logs');
    else if (tab === 'form') navigate('/form');
    else if (tab === 'analytics') navigate('/analytics');
    // Note: 'chat' and 'designer' are handled directly by Sidebar with custom onClick
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 flex flex-col">
        <header className="bg-white border-b border-gray-200 px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900">
            AI Booth Approval Assistant
          </h1>
          <p className="text-gray-600 mt-1">
            Streamline your booth approval process with AI-powered insights
          </p>
        </header>
        <div className="flex-1 overflow-hidden">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default MainLayout; 