import React from 'react';
import { Building, Users, Zap, Heart, Leaf, Crown, ChevronRight } from 'lucide-react';

interface PresetBooth {
  id: string;
  name: string;
  description: string;
  industry: string;
  dimensions: { width: number; depth: number; height: number };
  elements: Array<{
    type: string;
    position: [number, number, number];
    size: [number, number, number];
    price: number;
    label: string;
  }>;
  pricing: {
    materials: number;
    elements: number;
    lighting: number;
    setup: number;
    total: number;
  };
  image: string;
  icon: React.ReactNode;
  style: 'modern' | 'luxury' | 'tech' | 'eco' | 'traditional';
  colors: { primary: string; secondary: string; accent: string };
}

interface PresetBoothGalleryProps {
  onSelectPreset: (preset: PresetBooth) => void;
  onClose: () => void;
}

const PRESET_BOOTHS: PresetBooth[] = [
  {
    id: 'tech-innovation',
    name: 'Tech Innovation Hub',
    description: 'Perfect for technology companies showcasing AI, IoT, and digital solutions',
    industry: 'Technology & IT',
    dimensions: { width: 6, depth: 6, height: 3 },
    elements: [
      { type: 'interactive_kiosk', position: [0, 0, 0], size: [0.8, 1.6, 0.5], price: 11250, label: '📱 Interactive Kiosk' },
      { type: 'video_wall', position: [0, 0, -2.8], size: [4, 2.5, 0.2], price: 45000, label: '📺 Video Wall' },
      { type: 'hologram_projector', position: [2, 0, 2], size: [1, 1.5, 1], price: 30000, label: '👻 Hologram Projector' },
      { type: 'seating', position: [-2, 0, 2], size: [2, 0.8, 1.5], price: 3000, label: '🪑 Tech Lounge' },
      { type: 'reception_counter', position: [0, 0, 2.5], size: [3, 1.1, 0.8], price: 7500, label: '🏪 Reception' }
    ],
    pricing: { materials: 18720, elements: 96750, lighting: 10080, setup: 13500, total: 139050 },
    image: '/api/placeholder/300/200',
    icon: <Zap className="w-6 h-6" />,
    style: 'tech',
    colors: { primary: '#06B6D4', secondary: '#F3F4F6', accent: '#10B981' }
  },
  {
    id: 'healthcare-showcase',
    name: 'Healthcare Excellence',
    description: 'Designed for medical device manufacturers and healthcare innovators',
    industry: 'Healthcare & Medical',
    dimensions: { width: 8, depth: 6, height: 3 },
    elements: [
      { type: 'product_display', position: [0, 0, 0], size: [1.2, 1.8, 0.8], price: 5625, label: '📦 Medical Equipment Display' },
      { type: 'seating', position: [2.5, 0, 2], size: [2, 0.8, 1.5], price: 3000, label: '🪑 Consultation Area' },
      { type: 'reception_counter', position: [-2.5, 0, -2.5], size: [3, 1.1, 0.8], price: 7500, label: '🏪 Information Counter' },
      { type: 'interactive_kiosk', position: [-3, 0, 0], size: [0.8, 1.6, 0.5], price: 11250, label: '📱 Patient Info Kiosk' },
      { type: 'banner_stand', position: [3.5, 0, -2.5], size: [2, 3, 0.1], price: 1500, label: '🏁 Health Banner' }
    ],
    pricing: { materials: 24960, elements: 28875, lighting: 13440, setup: 18000, total: 85275 },
    image: '/api/placeholder/300/200',
    icon: <Heart className="w-6 h-6" />,
    style: 'modern',
    colors: { primary: '#3B82F6', secondary: '#F3F4F6', accent: '#EF4444' }
  },
  {
    id: 'luxury-automotive',
    name: 'Luxury Automotive',
    description: 'Premium showcase for high-end automotive brands and luxury vehicles',
    industry: 'Automotive',
    dimensions: { width: 10, depth: 8, height: 3.5 },
    elements: [
      { type: 'product_display', position: [0, 0, 0], size: [4, 1.5, 2], price: 18750, label: '🚗 Vehicle Display Platform' },
      { type: 'seating', position: [-3.5, 0, 3], size: [2.5, 0.8, 1.8], price: 8250, label: '🪑 VIP Lounge' },
      { type: 'coffee_bar', position: [3.5, 0, 3], size: [2.5, 1.1, 1], price: 10500, label: '☕ Executive Refreshments' },
      { type: 'video_wall', position: [0, 0, -3.8], size: [6, 2.8, 0.2], price: 67500, label: '📺 Brand Story Wall' },
      { type: 'reception_counter', position: [-4, 0, -2], size: [3, 1.1, 0.8], price: 7500, label: '🏪 Concierge Desk' }
    ],
    pricing: { materials: 41600, elements: 112500, lighting: 22400, setup: 30000, total: 206500 },
    image: '/api/placeholder/300/200',
    icon: <Crown className="w-6 h-6" />,
    style: 'luxury',
    colors: { primary: '#7C3AED', secondary: '#F8FAFC', accent: '#F59E0B' }
  },
  {
    id: 'sustainable-energy',
    name: 'Sustainable Energy',
    description: 'Eco-friendly design for renewable energy and sustainability companies',
    industry: 'Energy & Utilities',
    dimensions: { width: 7, depth: 7, height: 3 },
    elements: [
      { type: 'product_display', position: [0, 0, 0], size: [1.5, 2, 1], price: 7500, label: '🔋 Solar Panel Display' },
      { type: 'interactive_kiosk', position: [-2.5, 0, -2.5], size: [0.8, 1.6, 0.5], price: 11250, label: '📱 Carbon Calculator' },
      { type: 'seating', position: [2.5, 0, 2.5], size: [2, 0.8, 1.5], price: 3000, label: '🪑 Eco Seating' },
      { type: 'banner_stand', position: [0, 0, -3.2], size: [3, 3, 0.1], price: 2250, label: '🏁 Sustainability Banner' },
      { type: 'coffee_bar', position: [-3, 0, 1.5], size: [2, 1.1, 1], price: 10500, label: '☕ Organic Refreshments' }
    ],
    pricing: { materials: 25480, elements: 34500, lighting: 13720, setup: 18375, total: 92075 },
    image: '/api/placeholder/300/200',
    icon: <Leaf className="w-6 h-6" />,
    style: 'eco',
    colors: { primary: '#10B981', secondary: '#F3F4F6', accent: '#F59E0B' }
  },
  {
    id: 'fintech-hub',
    name: 'Fintech Innovation Hub',
    description: 'Modern financial services and blockchain technology showcase',
    industry: 'Financial Services',
    dimensions: { width: 6, depth: 8, height: 3 },
    elements: [
      { type: 'interactive_kiosk', position: [0, 0, 1], size: [0.8, 1.6, 0.5], price: 11250, label: '📱 Crypto Demo Station' },
      { type: 'video_wall', position: [0, 0, -3.8], size: [5, 2.5, 0.2], price: 56250, label: '📺 Market Data Wall' },
      { type: 'seating', position: [-2, 0, 3], size: [2, 0.8, 1.5], price: 3000, label: '🪑 Investor Lounge' },
      { type: 'reception_counter', position: [2.5, 0, 2], size: [2.5, 1.1, 0.8], price: 6750, label: '🏪 Advisory Desk' },
      { type: 'hologram_projector', position: [-2, 0, 0], size: [1, 1.5, 1], price: 30000, label: '👻 3D Charts Projector' }
    ],
    pricing: { materials: 24960, elements: 107250, lighting: 13440, setup: 18000, total: 163650 },
    image: '/api/placeholder/300/200',
    icon: <Building className="w-6 h-6" />,
    style: 'modern',
    colors: { primary: '#1E40AF', secondary: '#F3F4F6', accent: '#10B981' }
  },
  {
    id: 'saudi-hospitality',
    name: 'Saudi Hospitality Majlis',
    description: 'Traditional Saudi design with modern amenities for hospitality sector',
    industry: 'Tourism & Hospitality',
    dimensions: { width: 9, depth: 6, height: 3 },
    elements: [
      { type: 'majlis_seating', position: [0, 0, 1], size: [3, 0.6, 2], price: 12375, label: '🪑 Traditional Majlis' },
      { type: 'tea_station', position: [-3.5, 0, -2], size: [2, 1.1, 1], price: 6750, label: '🫖 Arabic Tea Service' },
      { type: 'banner_stand', position: [4, 0, -2.8], size: [2.5, 3, 0.1], price: 1875, label: '🏁 Arabic Heritage Banner' },
      { type: 'reception_counter', position: [2.5, 0, 2.5], size: [3, 1.1, 0.8], price: 7500, label: '🏪 Welcome Desk' },
      { type: 'prayer_area', position: [-3.5, 0, 2], size: [1.5, 0.1, 1.5], price: 2250, label: '🕌 Prayer Corner' }
    ],
    pricing: { materials: 28080, elements: 30750, lighting: 15120, setup: 20250, total: 94200 },
    image: '/api/placeholder/300/200',
    icon: <Users className="w-6 h-6" />,
    style: 'traditional',
    colors: { primary: '#92400E', secondary: '#FEF3C7', accent: '#059669' }
  }
];

const PresetBoothGallery: React.FC<PresetBoothGalleryProps> = ({ onSelectPreset, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-7xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold text-gray-800">Saudi Booth Preset Gallery</h2>
              <p className="text-gray-600 mt-1">Choose from professionally designed booth templates optimized for Saudi exhibitions</p>
            </div>
            <button 
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
            >
              ×
            </button>
          </div>
        </div>

        {/* Gallery Grid */}
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {PRESET_BOOTHS.map((booth) => (
              <div
                key={booth.id}
                className="bg-white rounded-xl border border-gray-200 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 cursor-pointer group"
                onClick={() => onSelectPreset(booth)}
              >
                {/* Image Placeholder */}
                <div className="h-48 bg-gradient-to-br from-gray-100 to-gray-200 rounded-t-xl relative overflow-hidden">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform duration-300">
                        {booth.icon}
                      </div>
                      <div className="text-sm text-gray-500 font-medium">{booth.dimensions.width}m × {booth.dimensions.depth}m</div>
                    </div>
                  </div>
                  
                  {/* Style Badge */}
                  <div className="absolute top-3 right-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      booth.style === 'tech' ? 'bg-cyan-100 text-cyan-700' :
                      booth.style === 'luxury' ? 'bg-purple-100 text-purple-700' :
                      booth.style === 'eco' ? 'bg-green-100 text-green-700' :
                      booth.style === 'traditional' ? 'bg-amber-100 text-amber-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {booth.style}
                    </span>
                  </div>
                </div>

                {/* Content */}
                <div className="p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-bold text-lg text-gray-800 group-hover:text-blue-600 transition-colors">
                        {booth.name}
                      </h3>
                      <p className="text-sm text-gray-500 font-medium">{booth.industry}</p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 group-hover:translate-x-1 transition-all duration-300" />
                  </div>

                  <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                    {booth.description}
                  </p>

                  {/* Features */}
                  <div className="space-y-2 mb-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">Elements:</span>
                      <span className="font-medium text-gray-700">{booth.elements.length} items</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">Size:</span>
                      <span className="font-medium text-gray-700">{booth.dimensions.width}m × {booth.dimensions.depth}m</span>
                    </div>
                  </div>

                  {/* Pricing */}
                  <div className="border-t pt-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-500">Total Cost:</span>
                      <span className="text-xl font-bold text-green-600">
                        SAR {booth.pricing.total.toLocaleString()}
                      </span>
                    </div>
                  </div>

                  {/* Action */}
                  <div className="mt-4">
                    <button className="w-full py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-300 font-medium group-hover:shadow-lg">
                      Select This Design
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 bg-gray-50">
          <div className="text-center">
            <p className="text-gray-600 text-sm">
              All designs include Saudi cultural considerations, Vision 2030 alignment, and professional setup.
            </p>
            <p className="text-gray-500 text-xs mt-1">
              Pricing in Saudi Riyals (SAR) includes materials, elements, lighting, and setup costs.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PresetBoothGallery;
