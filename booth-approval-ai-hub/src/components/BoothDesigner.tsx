import React, { useState, useCallback, useEffect } from 'react';
import { Save, Download, RotateCcw, Eye, Grid3x3, Settings, Plus, Palette } from 'lucide-react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { API_ENDPOINTS } from '@/lib/api';
import Booth3DViewer from './Booth3DViewer';
import AIChat from './AIChat';
import DesignControls from './DesignControls';
import Sidebar from './Sidebar';
import PresetBoothGallery from './PresetBoothGallery';

interface BoothDesign {
  id?: string;
  name: string;
  category: 'tech' | 'healthcare' | 'finance' | 'retail' | 'automotive' | 'energy' | 'custom';
  dimensions: { width: number; depth: number; height: number; };
  materials: {
    floor: 'wood' | 'marble' | 'concrete' | 'carpet' | 'metal' | 'glass';
    walls: 'drywall' | 'glass' | 'metal' | 'fabric' | 'led_panel' | 'bamboo';
    ceiling: 'standard' | 'suspended' | 'exposed' | 'curved' | 'led_sky';
  };
  colors: { primary: string; secondary: string; accent: string; };
  branding: { companyName: string; logo?: string; tagline?: string; brandColors: string[]; };
  elements: Array<{
    id: string;
    type: string;
    position: [number, number, number];
    size: [number, number, number];
    rotation: [number, number, number];
    color: string;
    material: string;
    label: string;
    price: number;
    locked?: boolean;
  }>;
    lighting: {
    ambient: number;
    spotlights: Array<{
      position: [number, number, number];
      intensity: number;
      color: string;
      type: 'spot' | 'directional' | 'led_strip';
    }>;
  };
  pricing: { materials: number; elements: number; lighting: number; setup: number; total: number; };
  style: 'modern' | 'futuristic' | 'minimal' | 'luxury' | 'industrial' | 'eco';
}

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



// Advanced AI Processing
class AdvancedAI {
  static async processNaturalLanguage(input: string, currentDesign: BoothDesign, sessionId: string): Promise<{
    intent: string;
    entities: string[];
    designChanges: Partial<BoothDesign>;
    confidence: number;
    response: string;
    aiAnalysis: any;
  }> {
    try {
      const response = await fetch(API_ENDPOINTS.BOOTH_DESIGN_CHAT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: input,
          current_design: currentDesign
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.status === 'success') {
    const words = input.toLowerCase().split(' ');
    const entities = this.extractEntities(words);
    const intent = this.classifyIntent(words);
    const designChanges = await this.generateDesignChanges(intent, entities, currentDesign);
    
    return {
      intent,
      entities,
      designChanges,
            confidence: 0.85,
            response: data.response,
            aiAnalysis: { sentiment: 'neutral', complexity: 'medium', intent, entities }
          };
        }
      }
    } catch (error) {
      console.error('AI Backend error:', error);
    }

    // Fallback
    return {
      intent: 'question',
      entities: [],
      designChanges: {},
      confidence: 0.6,
      response: "I'll help you with your booth design. Could you be more specific about what you'd like to change?",
      aiAnalysis: { sentiment: 'neutral', complexity: 'simple', intent: 'question', entities: [] }
    };
  }

  static extractEntities(words: string[]): string[] {
    const entityMap = {
      colors: ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white'],
      materials: ['wood', 'marble', 'metal', 'glass', 'concrete', 'fabric'],
      elements: ['counter', 'display', 'screen', 'seating', 'table', 'hologram'],
      tech: ['vr', 'ar', 'interactive', 'digital', 'hologram', 'ai']
    };
    
    const entities: string[] = [];
    words.forEach(word => {
      Object.values(entityMap).forEach(category => {
        if (category.includes(word)) {
          entities.push(word);
        }
      });
    });
    
    return [...new Set(entities)];
  }

  static classifyIntent(words: string[]): string {
    const intentKeywords = {
      design: ['create', 'design', 'make', 'build', 'generate'],
      modify: ['change', 'update', 'modify', 'adjust', 'edit'],
      question: ['what', 'how', 'why', 'when', 'where'],
      add: ['add', 'include', 'insert', 'place'],
      remove: ['remove', 'delete', 'take', 'eliminate']
    };
    
    for (const [intent, keywords] of Object.entries(intentKeywords)) {
      if (keywords.some(keyword => words.includes(keyword))) {
        return intent;
      }
    }
    
    return 'question';
  }

  static async generateDesignChanges(intent: string, entities: string[], currentDesign: BoothDesign): Promise<Partial<BoothDesign>> {
    const changes: Partial<BoothDesign> = {};
    
    if (intent === 'add') {
      if (entities.includes('hologram')) {
        changes.elements = [
          ...currentDesign.elements,
          {
            id: Date.now().toString(),
            type: 'hologram_projector',
            position: [2, 0, 2],
            size: [1, 1.5, 1],
            rotation: [0, 0, 0],
            color: '#06B6D4',
            material: 'metal',
            label: '👻 Hologram Projector',
            price: 8000
          }
        ];
      }
      if (entities.includes('screen')) {
        changes.elements = [
          ...currentDesign.elements,
          {
      id: Date.now().toString(),
            type: 'video_wall',
            position: [1, 0, 0.1],
            size: [4, 2.5, 0.2],
            rotation: [0, 0, 0],
            color: '#0F172A',
            material: 'metal',
            label: '📺 Video Wall',
            price: 12000
          }
        ];
      }
    }
    
    return changes;
  }
}

const BoothDesigner: React.FC = () => {
  const { session_id } = useParams<{ session_id: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Simplified state management
  const [messages, setMessages] = useState<DesignMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [showPresetGallery, setShowPresetGallery] = useState(false);
  
  const [design, setDesign] = useState<BoothDesign>({
    name: 'Custom Design',
    category: 'custom',
    style: 'modern',
    dimensions: { width: 6, depth: 6, height: 3 },
    materials: { floor: 'wood', walls: 'drywall', ceiling: 'standard' },
    colors: { primary: '#3B82F6', secondary: '#F3F4F6', accent: '#10B981' },
    branding: { companyName: 'Your Company', brandColors: [] },
    elements: [],
    lighting: { ambient: 0.4, spotlights: [] },
    pricing: { materials: 18750, elements: 0, lighting: 7500, setup: 11250, total: 37500 } // SAR pricing
  });

    // Load session data and chat history
  useEffect(() => {
    const loadSessionData = async () => {
      if (session_id) {
        try {
          // Load chat history for this session
          const historyResponse = await fetch(API_ENDPOINTS.CHAT_HISTORY_BY_ID(session_id));
          if (historyResponse.ok) {
            const chatHistory = await historyResponse.json();
            const convertedMessages = chatHistory.map((msg: any) => ({
              id: msg.id.toString(),
              role: msg.role === 'bot' ? 'ai_brain' : msg.role,
              content: msg.message,
              timestamp: new Date(msg.timestamp),
              aiGenerated: msg.role === 'bot',
              confidence: 0.85
            }));
            setMessages(convertedMessages);
          }

          // Try to load saved design for this session
          const designResponse = await fetch(API_ENDPOINTS.BOOTH_DESIGN_LOAD(session_id));
          if (designResponse.ok) {
            const savedDesign = await designResponse.json();
            if (savedDesign.design_data) {
              setDesign(savedDesign.design_data);
            }
        }
      } catch (error) {
          console.error('Error loading session data:', error);
        }
      }
    };

    if (location.state?.boothRequest && location.state?.shouldGeneratePreset) {
      // Coming from form - generate new design
      const requestData = location.state.boothRequest;
      const presetDesign = generatePresetDesign(requestData);
      setDesign(presetDesign);
      
      // Add welcome message from AI
    const welcomeMessage: DesignMessage = {
        id: Date.now().toString(),
        role: 'ai_brain',
        content: `🇸🇦 **Welcome ${requestData.company_name}!** I've created a custom booth design for **${requestData.event_name}**. This ${presetDesign.dimensions.width}m × ${presetDesign.dimensions.depth}m booth is perfect for ${requestData.industry} and fits your **SAR ${requestData.budget_inr?.toLocaleString()}** budget.\n\n✨ **Features included:**\n• Smart element placement\n• Saudi cultural considerations\n• Vision 2030 alignment\n• Professional Arabic signage ready\n\nHow would you like to customize it further?`,
          timestamp: new Date(),
          aiGenerated: true,
          confidence: 0.95,
        suggestions: ['Add Arabic signage', 'Include prayer area', 'Adjust colors', 'Add more tech elements']
      };
      
        setMessages([welcomeMessage]);
    } else {
      // Existing session - load data
      loadSessionData();
    }
  }, [location.state, session_id]);

  // AI Chat Handler with Design Context
  // Handle finalize command for screenshot and report generation
  const handleFinalize = useCallback(async () => {
    try {
      // Take screenshot of the 3D canvas
      const canvas = document.querySelector('canvas');
      if (!canvas) {
        throw new Error('Canvas not found');
      }

      // Convert canvas to image
      const screenshotDataUrl = canvas.toDataURL('image/png');
      
      // Generate final report
      const reportData = {
        session_id: session_id || Date.now().toString(),
        design_name: design.name,
        design_data: design,
        screenshot: screenshotDataUrl,
        timestamp: new Date().toISOString(),
        pricing: design.pricing,
        elements_count: design.elements.length,
        booth_dimensions: `${design.dimensions.width}m × ${design.dimensions.depth}m × ${design.dimensions.height}m`,
        total_cost_sar: design.pricing.total
      };

      // Send to backend for report generation
      const response = await fetch(API_ENDPOINTS.FINALIZE_DESIGN, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportData)
      });

      if (response.ok) {
        const result = await response.json();
        
        // Show success message
        const successMessage: DesignMessage = {
          id: Date.now().toString(),
          role: 'ai_brain',
          content: `🎉 **Design Finalized Successfully!**\n\n📸 **Screenshot captured** and saved to your portfolio\n📋 **Final report generated** with complete specifications\n💰 **Total Cost: SAR ${design.pricing.total.toLocaleString()}**\n\n✅ Your booth design "${design.name}" is now ready for approval review!\n\n**Report ID:** ${result.report_id}`,
      timestamp: new Date(),
        aiGenerated: true,
          confidence: 1.0
        };
        
        setMessages(prev => [...prev, successMessage]);
      } else {
        throw new Error('Failed to generate report');
      }
    } catch (error) {
      console.error('Finalize error:', error);
      const errorMessage: DesignMessage = {
        id: Date.now().toString(),
        role: 'system',
        content: `❌ **Finalization Failed**: ${error.message || 'Unable to finalize design'}. Please try again.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [design, session_id]);

  const handleSendMessage = useCallback(async (message: string, isVoiceInput = false) => {
    // Check for finalize command
    if (message.toLowerCase().includes('finalize') || message.toLowerCase().includes('finalise')) {
      await handleFinalize();
      return;
    }

    const currentSessionId = session_id || Date.now().toString();
    
    const userMessage: DesignMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      voiceInput: isVoiceInput
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

        try {
      // Save user message to backend
      await fetch(API_ENDPOINTS.CHAT_SAVE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          role: 'user',
          message: message
        })
      });

      // Save current design before AI processing
      await fetch(API_ENDPOINTS.BOOTH_DESIGN_SAVE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          design_name: design.name,
          design_data: design
        })
      });

      const aiResult = await AdvancedAI.processNaturalLanguage(message, design, currentSessionId);
      
      // Create AI response with smart suggestions and better formatting
      const smartSuggestions = generateSmartSuggestions(design, message);
      const aiMessage: DesignMessage = {
        id: (Date.now() + 1).toString(),
        role: 'ai_brain',
        content: aiResult.response,
        timestamp: new Date(),
        aiGenerated: true,
        confidence: aiResult.confidence,
        suggestions: smartSuggestions
      };

      setMessages(prev => [...prev, aiMessage]);
      
      // Save AI response to backend
      await fetch(API_ENDPOINTS.CHAT_SAVE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          role: 'bot',
          message: aiResult.response
        })
      });

      // Apply design changes and save updated design
      if (Object.keys(aiResult.designChanges).length > 0) {
        const updatedDesign = { ...design, ...aiResult.designChanges };
        setDesign(updatedDesign);
        
        // Save updated design
        await fetch(API_ENDPOINTS.BOOTH_DESIGN_SAVE, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: currentSessionId,
            design_name: updatedDesign.name,
            design_data: updatedDesign
          })
        });
      }
      
    } catch (error) {
      console.error('Error processing message:', error);
      
      const errorMessage: DesignMessage = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: `🚨 **Connection Issue**: I'm having trouble connecting to the AI backend. Please ensure:\n\n• Backend server is running on http://localhost:8000\n• Database is accessible\n• No network connectivity issues\n\n**Error details:** ${error instanceof Error ? error.message : 'Unknown error'}\n\nTry refreshing the page or contact support if this persists.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [design, session_id]);

  // Generate Smart Suggestions based on current design and user input
  const generateSmartSuggestions = (currentDesign: BoothDesign, userMessage: string) => {
    const suggestions = [];
    
    if (currentDesign.elements.length === 0) {
      suggestions.push("Add interactive displays", "Add seating area", "Add product showcase");
    } else if (currentDesign.elements.length < 3) {
      suggestions.push("Add hologram projector", "Add video wall", "Add coffee station");
    } else {
      suggestions.push("Optimize layout", "Change color scheme", "Add lighting effects");
    }
    
    // Context-aware suggestions
    if (userMessage.toLowerCase().includes('tech')) {
      suggestions.push("Add VR station", "Add AI demo area");
    }
    if (userMessage.toLowerCase().includes('saudi')) {
      suggestions.push("Add Arabic signage", "Use local materials");
    }
    
    return suggestions.slice(0, 4);
  };

  // Design update handler
  const handleDesignUpdate = useCallback((updates: Partial<BoothDesign>) => {
    setDesign(prev => ({ ...prev, ...updates }));
  }, []);

  // Smart element placement within booth boundaries with strict overflow prevention
  const getSmartPosition = (elementSize: [number, number, number], existingElements: any[]) => {
    const boothWidth = design.dimensions.width;
    const boothDepth = design.dimensions.depth;
    
    // Strict boundary checking - ensure element never goes outside booth
    const maxX = (boothWidth / 2) - (elementSize[0] / 2) - 0.3; // 30cm buffer from walls
    const maxZ = (boothDepth / 2) - (elementSize[2] / 2) - 0.3;
    
    // Clamp positions to valid range
    const clampPosition = (pos: [number, number, number]): [number, number, number] => {
      return [
        Math.max(-maxX, Math.min(maxX, pos[0])),
        pos[1],
        Math.max(-maxZ, Math.min(maxZ, pos[2]))
      ];
    };
    
    // Grid-based positioning for better organization
    const gridSize = 1.0; // 1 meter grid
    const gridPositions: [number, number, number][] = [];
    
    for (let x = -maxX; x <= maxX; x += gridSize) {
      for (let z = -maxZ; z <= maxZ; z += gridSize) {
        gridPositions.push([x, 0, z]);
      }
    }
    
    // Sort by distance from center (prefer center positions)
    gridPositions.sort((a, b) => {
      const distA = Math.sqrt(a[0] * a[0] + a[2] * a[2]);
      const distB = Math.sqrt(b[0] * b[0] + b[2] * b[2]);
      return distA - distB;
    });
    
    // Find first position without collision
    for (const pos of gridPositions) {
      let hasCollision = false;
      
      // Check element boundaries don't exceed booth
      const elementLeft = pos[0] - elementSize[0] / 2;
      const elementRight = pos[0] + elementSize[0] / 2;
      const elementFront = pos[2] + elementSize[2] / 2;
      const elementBack = pos[2] - elementSize[2] / 2;
      
      if (elementLeft < -boothWidth/2 + 0.1 || elementRight > boothWidth/2 - 0.1 ||
          elementFront > boothDepth/2 - 0.1 || elementBack < -boothDepth/2 + 0.1) {
        continue; // Skip positions that would cause overflow
      }
      
      // Check collision with existing elements
      for (const existing of existingElements) {
        const dx = Math.abs(pos[0] - existing.position[0]);
        const dz = Math.abs(pos[2] - existing.position[2]);
        const minDistX = (elementSize[0] + existing.size[0]) / 2 + 0.5; // 50cm buffer
        const minDistZ = (elementSize[2] + existing.size[2]) / 2 + 0.5;
        
        if (dx < minDistX && dz < minDistZ) {
          hasCollision = true;
          break;
        }
      }
      
      if (!hasCollision) {
        return clampPosition(pos);
      }
    }
    
    // If no free grid position, place at center but ensure it doesn't overflow
    return clampPosition([0, 0, 0]);
  };

  // Add element handler with Saudi pricing and smart positioning
  const handleAddElement = useCallback((elementType: string) => {
    // Saudi pricing in SAR (approximate 3.75x USD conversion + local market rates)
    const elementData = {
      interactive_kiosk: { size: [0.8, 1.6, 0.5], price: 11250, label: '📱 Interactive Kiosk' },
      hologram_projector: { size: [1, 1.5, 1], price: 30000, label: '👻 Hologram Projector' },
      video_wall: { size: [Math.min(4, design.dimensions.width * 0.8), 2.5, 0.2], price: 45000, label: '📺 Video Wall' },
      seating: { size: [2, 0.8, 1.5], price: 3000, label: '🪑 Lounge Seating' },
      coffee_bar: { size: [2.5, 1.1, 1], price: 10500, label: '☕ Coffee Bar' },
      vr_station: { size: [1.5, 2, 1.5], price: 18750, label: '🥽 VR Station' },
      product_display: { size: [1.2, 1.8, 0.8], price: 5625, label: '📦 Product Display' },
      banner_stand: { size: [2, 3, 0.1], price: 1500, label: '🏁 Banner Stand' },
      reception_counter: { size: [Math.min(3, design.dimensions.width * 0.6), 1.1, 0.8], price: 7500, label: '🏪 Reception Counter' }
    };

    const data = elementData[elementType as keyof typeof elementData] || elementData.interactive_kiosk;
    
    // Smart positioning
    const smartPosition = getSmartPosition(data.size as [number, number, number], design.elements);
    
    const newElement = {
        id: Date.now().toString(),
      type: elementType,
      position: smartPosition as [number, number, number],
      size: data.size as [number, number, number],
      rotation: [0, 0, 0] as [number, number, number],
      color: design.colors.primary,
      material: 'metal',
      label: data.label,
      price: data.price
    };
    
    setDesign(prev => ({
        ...prev,
      elements: [...prev.elements, newElement]
    }));
  }, [design.elements, design.dimensions, design.colors.primary]);

  // Remove element handler
  const handleRemoveElement = useCallback((elementId: string) => {
    setDesign(prev => ({
          ...prev,
      elements: prev.elements.filter(el => el.id !== elementId)
    }));
  }, []);

  // Voice handlers
  const handleVoiceCommand = useCallback(() => {
    if (!isVoiceEnabled) return;

    // Check if speech recognition is supported
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      const errorMessage: DesignMessage = {
        id: Date.now().toString(),
        role: 'system',
        content: '🎤 **Voice Recognition Not Supported**: Your browser doesn\'t support speech recognition. Please use a modern browser like Chrome.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    try {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      const listeningMessage: DesignMessage = {
      id: Date.now().toString(),
      role: 'system',
        content: '🎤 **Listening...** Speak your booth design question or command.',
      timestamp: new Date()
    };
      setMessages(prev => [...prev, listeningMessage]);

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        console.log('Voice input:', transcript);
        
        // Send the voice input to the chat handler
        handleSendMessage(transcript, true);
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        const errorMessage: DesignMessage = {
      id: Date.now().toString(),
      role: 'system',
          content: `🎤 **Voice Error**: ${event.error}. Please try again or type your message.`,
      timestamp: new Date()
    };
        setMessages(prev => [...prev, errorMessage]);
      };

      recognition.onend = () => {
        console.log('Speech recognition ended');
      };

      recognition.start();
    } catch (error) {
      console.error('Voice recognition setup error:', error);
      const errorMessage: DesignMessage = {
                            id: Date.now().toString(),
        role: 'system',
        content: '🎤 **Voice Setup Error**: Unable to start voice recognition. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [isVoiceEnabled, handleSendMessage]);

    const handleToggleVoice = useCallback(() => {
    setIsVoiceEnabled(prev => !prev);
  }, []);

  // Save design functionality
  const handleSaveDesign = useCallback(async () => {
    try {
      const currentSessionId = session_id || Date.now().toString();
      
      const response = await fetch(API_ENDPOINTS.BOOTH_DESIGN_SAVE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          design_name: design.name,
          design_data: design
        })
      });

      if (response.ok) {
        // Show success message to user
        const successMessage: DesignMessage = {
        id: Date.now().toString(),
          role: 'system',
          content: `✅ **Design Saved Successfully!**\n\nYour booth design "${design.name}" has been saved to session ${currentSessionId}. You can access it anytime by returning to this session.`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, successMessage]);
      } else {
        throw new Error('Failed to save design');
      }
    } catch (error) {
      console.error('Error saving design:', error);
      const errorMessage: DesignMessage = {
        id: Date.now().toString(),
        role: 'system',
        content: `❌ **Save Failed**: Unable to save your design. Please check the backend connection and try again.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [design, session_id]);

  // Handle element position change for drag functionality
  const handleElementPositionChange = useCallback((elementId: string, newPosition: [number, number, number]) => {
    setDesign(prev => ({
      ...prev,
      elements: prev.elements.map(el => 
        el.id === elementId 
          ? { ...el, position: newPosition }
          : el
      )
    }));
  }, []);

  // Handle preset booth selection
  const handlePresetSelection = (preset: any) => {
    const newDesign: BoothDesign = {
      name: preset.name,
      category: preset.industry === 'Technology & IT' ? 'tech' : 
                preset.industry === 'Healthcare & Medical' ? 'healthcare' :
                preset.industry === 'Financial Services' ? 'finance' :
                preset.industry === 'Automotive' ? 'automotive' :
                preset.industry === 'Energy & Utilities' ? 'energy' : 'custom',
      style: preset.style,
      dimensions: preset.dimensions,
      materials: { 
        floor: preset.style === 'luxury' ? 'marble' : preset.style === 'eco' ? 'wood' : 'wood',
        walls: preset.style === 'tech' ? 'led_panel' : 'drywall',
        ceiling: preset.style === 'luxury' ? 'suspended' : 'standard'
      },
      colors: preset.colors,
      branding: { companyName: preset.name.split(' ')[0], brandColors: [] },
      elements: preset.elements.map((el: any) => ({
        ...el,
        id: `preset-${Date.now()}-${Math.random()}`,
        rotation: [0, 0, 0] as [number, number, number],
        color: preset.colors.primary,
        material: 'metal',
        locked: false
      })),
      lighting: { ambient: 0.5, spotlights: [] },
      pricing: preset.pricing
    };

    setDesign(newDesign);
    setShowPresetGallery(false);

    // Add AI message about the preset
    const presetMessage: DesignMessage = {
        id: Date.now().toString(),
        role: 'ai_brain',
      content: `🎨 **Perfect Choice!** I've loaded the **${preset.name}** design for you.\n\n✨ **This ${preset.industry} booth includes:**\n• ${preset.elements.length} professionally positioned elements\n• ${preset.dimensions.width}m × ${preset.dimensions.depth}m optimized layout\n• Saudi cultural considerations built-in\n• **Total cost: SAR ${preset.pricing.total.toLocaleString()}**\n\nFeel free to customize any element or ask me to make adjustments!`,
        timestamp: new Date(),
        aiGenerated: true,
        confidence: 0.95,
      suggestions: ['Adjust colors', 'Add elements', 'Modify layout', 'Add Arabic signage']
    };

    setMessages(prev => [...prev, presetMessage]);
  };

  // Generate preset design based on booth request
  const generatePresetDesign = (requestData: any) => {
    const { booth_size_sqm, preferred_style, industry, budget_inr } = requestData;
    const width = Math.sqrt(booth_size_sqm);
    const depth = Math.sqrt(booth_size_sqm);
    
    // Calculate smart element placement within booth dimensions
    const maxX = width / 2 - 0.5;
    const maxZ = depth / 2 - 0.5;
    
    let presetElements = [];
    
    // Industry-specific element suggestions
    switch (industry) {
      case 'Technology & IT':
        presetElements = [
          { type: 'interactive_kiosk', position: [0, 0, 0], size: [0.8, 1.6, 0.5], price: 11250, label: '📱 Interactive Kiosk' },
          { type: 'video_wall', position: [0, 0, -maxZ], size: [Math.min(4, width), 2.5, 0.2], price: 45000, label: '📺 Video Wall' },
          { type: 'hologram_projector', position: [maxX * 0.7, 0, maxZ * 0.7], size: [1, 1.5, 1], price: 30000, label: '👻 Hologram Projector' }
        ];
        break;
      case 'Healthcare & Medical':
        presetElements = [
          { type: 'product_display', position: [0, 0, 0], size: [1.2, 1.8, 0.8], price: 5625, label: '📦 Medical Equipment Display' },
          { type: 'seating', position: [maxX * 0.8, 0, maxZ * 0.8], size: [2, 0.8, 1.5], price: 3000, label: '🪑 Consultation Area' },
          { type: 'reception_counter', position: [0, 0, maxZ * 0.9], size: [3, 1.1, 0.8], price: 7500, label: '🏪 Information Counter' }
        ];
        break;
      default:
        presetElements = [
          { type: 'reception_counter', position: [0, 0, -maxZ * 0.8], size: [3, 1.1, 0.8], price: 7500, label: '🏪 Reception Counter' },
          { type: 'product_display', position: [maxX * 0.6, 0, 0], size: [1.2, 1.8, 0.8], price: 5625, label: '📦 Product Display' },
          { type: 'seating', position: [-maxX * 0.6, 0, maxZ * 0.6], size: [2, 0.8, 1.5], price: 3000, label: '🪑 Meeting Area' }
        ];
    }

    // Add ID and smart positioning
    presetElements = presetElements.map((el, index) => ({
      ...el,
      id: `preset-${Date.now()}-${index}`,
      rotation: [0, 0, 0] as [number, number, number],
      color: '#3B82F6',
      material: 'metal',
      locked: false
    }));

    // Generate new design
    const presetDesign: BoothDesign = {
      name: `${requestData.company_name} - ${requestData.event_name}`,
      category: industry === 'Technology & IT' ? 'tech' : 'custom',
      style: preferred_style || 'modern',
      dimensions: { width: Math.ceil(width), depth: Math.ceil(depth), height: 3 },
      materials: { 
        floor: preferred_style === 'luxury' ? 'marble' : 'wood', 
        walls: preferred_style === 'tech' ? 'led_panel' : 'drywall', 
        ceiling: preferred_style === 'luxury' ? 'suspended' : 'standard' 
      },
      colors: { 
        primary: preferred_style === 'tech' ? '#06B6D4' : '#3B82F6', 
        secondary: '#F3F4F6', 
        accent: '#10B981' 
      },
      branding: { 
        companyName: requestData.company_name, 
        brandColors: [] 
      },
      elements: presetElements,
      lighting: { ambient: 0.5, spotlights: [] },
      pricing: { 
        materials: Math.round(booth_size_sqm * 520), 
        elements: presetElements.reduce((sum, el) => sum + el.price, 0), 
        lighting: Math.round(booth_size_sqm * 280), 
        setup: Math.round(booth_size_sqm * 375), 
        total: 0 
      }
    };

    // Calculate total pricing
    presetDesign.pricing.total = presetDesign.pricing.materials + presetDesign.pricing.elements + presetDesign.pricing.lighting + presetDesign.pricing.setup;

    return presetDesign;
  };

    return (
    <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 flex">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <div className="flex-1 ml-64 flex flex-col">
        {/* Header */}
        <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 px-8 py-6 shadow-lg">
                  <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-lg">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-800 via-blue-600 to-purple-600 bg-clip-text text-transparent">
                Booth Designer
                </h1>
                  </div>
            <div className="flex items-center space-x-4 px-4 py-2 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200/50">
              <Grid3x3 className="w-5 h-5 text-blue-600" />
              <span className="font-medium text-gray-700">{design.dimensions.width}m × {design.dimensions.depth}m</span>
              <span className="text-gray-400">•</span>
              <span className="font-bold text-green-600">SAR {design.pricing.total.toLocaleString()}</span>
                </div>
                      </div>
              
                                                            <div className="flex items-center space-x-3">
                <button
              onClick={() => setShowPresetGallery(true)}
              className="group px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl hover:from-purple-600 hover:to-pink-700 transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              <Palette className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" />
              <span className="font-medium">Preset Gallery</span>
                </button>
                <button
              onClick={() => navigate('/')}
              className="group px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              <Plus className="w-5 h-5 group-hover:rotate-90 transition-transform duration-300" />
              <span className="font-medium">New Booth</span>
                </button>
                <button
              onClick={handleSaveDesign}
              className="group px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              <Save className="w-5 h-5 group-hover:animate-pulse" />
              <span className="font-medium">Save</span>
            </button>
            <button className="group px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl hover:from-gray-700 hover:to-gray-800 transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105">
              <Download className="w-5 h-5 group-hover:animate-bounce" />
              <span className="font-medium">Export</span>
                </button>
                      </div>
                    </div>
                  </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* 3D Viewer */}
        <div className="flex-1 bg-gray-900">
          <Booth3DViewer
              design={design} 
              selectedElement={selectedElement}
              onElementSelect={setSelectedElement}
              onElementPositionChange={handleElementPositionChange}
            className="h-full"
          />
                    </div>

        {/* Side Panel */}
        <div className="w-96 bg-white/90 backdrop-blur-sm border-l border-gray-200/50 flex flex-col shadow-2xl">
          {/* Design Controls */}
          <div className="h-1/2 border-b border-gray-200/50">
            <DesignControls
              design={design}
              selectedElement={selectedElement}
              onDesignUpdate={handleDesignUpdate}
              onAddElement={handleAddElement}
              onRemoveElement={handleRemoveElement}
            />
          </div>

          {/* AI Chat */}
          <div className="h-1/2">
            <AIChat
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              onVoiceCommand={handleVoiceCommand}
              isVoiceEnabled={isVoiceEnabled}
              onToggleVoice={handleToggleVoice}
            />
          </div>
        </div>
      </div>
            </div>

      {/* Preset Booth Gallery Modal */}
      {showPresetGallery && (
        <PresetBoothGallery
          onSelectPreset={handlePresetSelection}
          onClose={() => setShowPresetGallery(false)}
        />
      )}
    </div>
  );
};

export default BoothDesigner;