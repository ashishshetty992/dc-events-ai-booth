import React from 'react';

const Index = () => (
  <div className="flex flex-col items-center justify-center h-full">
    <h2 className="text-3xl font-bold mb-4 text-gray-900">Welcome to the AI Booth Approval Assistant</h2>
    <p className="text-gray-600 max-w-xl text-center mb-8">
      Get started with AI-powered booth design and approval assistance. Submit a form for automatic booth generation, or chat directly with our AI assistant.
    </p>
    <div className="flex gap-4">
      <button 
        onClick={() => window.location.href = '/form'}
        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
      >
        🎨 Submit Booth Request
      </button>
      <button 
        onClick={() => window.location.href = `/chat/session_${Date.now()}`}
        className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
      >
        💬 AI Assistant
      </button>
    </div>
  </div>
);

export default Index;
