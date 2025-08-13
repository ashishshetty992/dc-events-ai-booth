import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, Mic, MicOff, Sparkles, Zap, MessageCircle, Lightbulb, Wand2 } from 'lucide-react';

interface DesignMessage {
  id: string;
  role: 'user' | 'designer' | 'ai_brain' | 'system';
  content: string;
  timestamp: Date;
  suggestions?: string[];
  aiGenerated?: boolean;
  confidence?: number;
  voiceInput?: boolean;
  imageUrl?: string;
}

interface AIChatProps {
  messages: DesignMessage[];
  onSendMessage: (message: string, isVoiceInput?: boolean) => Promise<void>;
  isLoading: boolean;
  onVoiceCommand: () => void;
  isVoiceEnabled: boolean;
  onToggleVoice: () => void;
}

const AIChat: React.FC<AIChatProps> = ({
  messages,
  onSendMessage,
  isLoading,
  onVoiceCommand,
  isVoiceEnabled,
  onToggleVoice
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;
    
    const message = inputMessage.trim();
    setInputMessage('');
    await onSendMessage(message);
  };

  const handleSuggestionClick = async (suggestion: string) => {
    await onSendMessage(suggestion);
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-4000"></div>
      </div>

      {/* Header */}
      <div className="relative z-10 p-6 border-b border-white/10 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center text-white text-2xl shadow-2xl">
                🤖
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white animate-pulse"></div>
            </div>
            <div>
              <h3 className="font-bold text-white text-lg">AI Design Assistant</h3>
              <p className="text-purple-200 text-sm flex items-center">
                <Sparkles className="w-3 h-3 mr-1" />
                Powered by Advanced AI
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={onToggleVoice}
              className={`p-3 rounded-xl transition-all duration-300 transform hover:scale-105 ${
                isVoiceEnabled 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 shadow-lg shadow-red-500/25' 
                  : 'bg-white/10 text-white hover:bg-white/20 backdrop-blur-sm'
              }`}
              title={isVoiceEnabled ? 'Stop Voice' : 'Start Voice'}
            >
              {isVoiceEnabled ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>
            <div className="px-3 py-2 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl backdrop-blur-sm border border-white/10">
              <span className="text-white text-sm font-medium">Online</span>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 min-h-0 relative z-10 custom-scrollbar">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <div className="relative mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 rounded-3xl flex items-center justify-center mx-auto shadow-2xl">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
              <div className="absolute -inset-4 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 rounded-3xl opacity-20 blur-lg animate-pulse"></div>
            </div>
            <h3 className="text-white font-bold text-2xl mb-3">🚀 AI Design Assistant</h3>
            <p className="text-purple-200 text-lg mb-8 max-w-md mx-auto leading-relaxed">
              I'm your intelligent booth design companion. Let's create something extraordinary together!
            </p>
            <div className="grid grid-cols-2 gap-3 max-w-lg mx-auto">
              {[
                { text: "Design a tech booth", icon: "🏗️" },
                { text: "Add holographic displays", icon: "👻" }, 
                { text: "Review my current design", icon: "🔍" },
                { text: "Suggest improvements", icon: "💡" }
              ].map((suggestion) => (
                <button
                  key={suggestion.text}
                  onClick={() => handleSuggestionClick(suggestion.text)}
                  className="group px-4 py-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl text-sm text-white hover:bg-white/20 transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
                >
                  <span className="text-lg mb-1 block group-hover:animate-bounce">{suggestion.icon}</span>
                  {suggestion.text}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeInUp`}
          >
            <div className={`max-w-[85%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
              {message.role !== 'user' && (
                <div className="flex items-center mb-2 ml-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center mr-2 shadow-lg">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-sm font-medium text-purple-200">
                    AI Assistant
                  </span>
                  {message.confidence && (
                    <span className="ml-2 px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded-full border border-green-500/30">
                      {Math.round(message.confidence * 100)}% confident
                    </span>
                  )}
                </div>
              )}
              
              <div
                className={`rounded-2xl px-5 py-4 shadow-lg backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white ml-4'
                    : 'bg-white/10 text-white border border-white/20 mr-4'
                }`}
              >
                <div className="prose prose-sm max-w-none text-inherit">
                  <div 
                    className="whitespace-pre-wrap leading-relaxed"
                    dangerouslySetInnerHTML={{ 
                      __html: message.content.replace(/\n/g, '<br/>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') 
                    }}
                  />
                </div>

                {message.suggestions && message.suggestions.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {message.suggestions.map((suggestion, index) => (
                      <button
                        key={index}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="group px-3 py-2 bg-white/20 hover:bg-white/30 border border-white/30 rounded-xl text-xs text-white transition-all duration-300 transform hover:scale-105"
                      >
                        <Lightbulb className="w-3 h-3 inline mr-1 group-hover:animate-pulse" />
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start animate-fadeInUp">
            <div className="flex items-center mb-2 ml-2">
              <div className="w-7 h-7 bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center mr-2 shadow-lg">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-medium text-purple-200">AI Assistant</span>
            </div>
            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl px-5 py-4 mr-4 shadow-lg">
              <div className="flex items-center space-x-3">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-purple-200">AI is thinking...</span>
                <Zap className="w-4 h-4 text-yellow-400 animate-pulse" />
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="relative z-10 p-6 border-t border-white/10 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="flex space-x-4">
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask me anything about your booth design..."
              className="w-full px-6 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent text-white placeholder-purple-200 transition-all duration-300"
              disabled={isLoading}
            />
            <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
              <MessageCircle className="w-5 h-5 text-purple-300" />
            </div>
          </div>
          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className="group px-6 py-4 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white rounded-2xl hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-3 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-2xl"
          >
            <Send className="w-5 h-5 group-hover:animate-pulse" />
            <span className="font-medium">Send</span>
            <Wand2 className="w-4 h-4 group-hover:rotate-12 transition-transform duration-300" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default AIChat;