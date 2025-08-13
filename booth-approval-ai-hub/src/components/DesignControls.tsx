import React from 'react';
import { Palette, Package, Ruler, Settings, Copy, Trash2, RotateCw } from 'lucide-react';

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

interface DesignControlsProps {
  design: BoothDesign;
  selectedElement: string | null;
  onDesignUpdate: (updates: Partial<BoothDesign>) => void;
  onAddElement: (elementType: string) => void;
  onRemoveElement: (elementId: string) => void;
}

const ELEMENT_LIBRARY = {
  'Technology & Media': [
    { type: 'screen', label: '📺 LED Screen', price: 9375 },
    { type: 'interactive_kiosk', label: '📱 Interactive Kiosk', price: 11250 },
    { type: 'hologram_projector', label: '👻 Hologram Projector', price: 30000 },
    { type: 'vr_station', label: '🥽 VR Station', price: 18750 },
    { type: 'video_wall', label: '📺 Video Wall', price: 45000 },
  ],
  'Furniture & Seating': [
    { type: 'seating', label: '🪑 Lounge Seating', price: 3000 },
    { type: 'reception_counter', label: '🏪 Reception Counter', price: 7500 },
    { type: 'conference_table', label: '📋 Conference Table', price: 4500 },
    { type: 'coffee_bar', label: '☕ Coffee Bar', price: 10500 },
  ],
  'Display & Showcase': [
    { type: 'product_display', label: '📦 Product Display', price: 5625 },
    { type: 'showcase_tower', label: '🏗️ Showcase Tower', price: 13125 },
    { type: 'glass_showcase', label: '💎 Glass Showcase', price: 10500 },
  ],
  'Saudi Specialties': [
    { type: 'banner_stand', label: '🏁 Arabic Banner', price: 1500 },
    { type: 'majlis_seating', label: '🪑 Majlis Seating', price: 8250 },
    { type: 'prayer_area', label: '🕌 Prayer Corner', price: 2250 },
    { type: 'tea_station', label: '🫖 Arabic Tea Station', price: 6750 },
  ]
};

const DesignControls: React.FC<DesignControlsProps> = ({
  design,
  selectedElement,
  onDesignUpdate,
  onAddElement,
  onRemoveElement
}) => {
  const [activeTab, setActiveTab] = React.useState<'elements' | 'style' | 'dimensions'>('elements');

  const selectedElementData = design.elements.find(el => el.id === selectedElement);

  const calculateTotalPrice = () => {
    const elementsTotal = design.elements.reduce((total, element) => total + (element.price || 0), 0);
    const area = design.dimensions.width * design.dimensions.depth;
    const materialsTotal = Math.round(area * 520); // SAR 520 per sqm for materials
    const lightingTotal = Math.round(area * 280); // SAR 280 per sqm for lighting
    const setupTotal = Math.round(area * 375); // SAR 375 per sqm for setup
    
    return {
      materials: materialsTotal,
      elements: elementsTotal,
      lighting: lightingTotal,
      setup: setupTotal,
      total: materialsTotal + elementsTotal + lightingTotal + setupTotal
    };
  };

  const updatePricing = () => {
    const newPricing = calculateTotalPrice();
    onDesignUpdate({ pricing: newPricing });
  };

  React.useEffect(() => {
    updatePricing();
  }, [design.elements]);

  const handleDimensionChange = (dimension: 'width' | 'depth' | 'height', value: number) => {
    onDesignUpdate({
      dimensions: {
        ...design.dimensions,
        [dimension]: Math.max(1, Math.min(20, value))
      }
    });
  };

  const handleColorChange = (colorType: 'primary' | 'secondary' | 'accent', color: string) => {
    onDesignUpdate({
      colors: {
        ...design.colors,
        [colorType]: color
      }
    });
  };

  const handleMaterialChange = (materialType: 'floor' | 'walls' | 'ceiling', material: string) => {
    onDesignUpdate({
      materials: {
        ...design.materials,
        [materialType]: material
      }
    });
  };

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 bg-gray-50">
        {[
          { id: 'elements', label: 'Elements', icon: Package },
          { id: 'style', label: 'Style', icon: Palette },
          { id: 'dimensions', label: 'Size', icon: Ruler }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as any)}
            className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
              activeTab === id
                ? 'text-blue-600 border-b-2 border-blue-600 bg-white'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <Icon className="w-4 h-4 inline mr-1" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'elements' && (
          <div className="space-y-4">
            {/* Selected Element Info */}
            {selectedElementData && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
                <h4 className="font-medium text-yellow-800 mb-2">Selected Element</h4>
                                  <div className="text-sm text-yellow-700">
                  <div className="font-medium">{selectedElementData.label}</div>
                  <div className="text-xs mt-1">SAR {selectedElementData.price?.toLocaleString()}</div>
                  <div className="flex space-x-2 mt-2">
                    <button
                      onClick={() => {
                        const updatedElements = design.elements.map(el =>
                          el.id === selectedElement
                            ? { ...el, rotation: [el.rotation[0], el.rotation[1] + Math.PI / 4, el.rotation[2]] as [number, number, number] }
                            : el
                        );
                        onDesignUpdate({ elements: updatedElements });
                      }}
                      className="px-2 py-1 bg-yellow-200 text-yellow-800 rounded text-xs hover:bg-yellow-300"
                    >
                      <RotateCw className="w-3 h-3 inline mr-1" />
                      Rotate
                    </button>
                    <button
                      onClick={() => onRemoveElement(selectedElement)}
                      className="px-2 py-1 bg-red-200 text-red-800 rounded text-xs hover:bg-red-300"
                    >
                      <Trash2 className="w-3 h-3 inline mr-1" />
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Element Library */}
            <div className="space-y-4">
              {Object.entries(ELEMENT_LIBRARY).map(([category, elements]) => (
                <div key={category}>
                  <h4 className="font-medium text-gray-800 mb-2 text-sm">{category}</h4>
                  <div className="grid grid-cols-1 gap-2">
                    {elements.map((element) => (
                      <button
                        key={element.type}
                        onClick={() => onAddElement(element.type)}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-left"
                      >
                        <div>
                          <div className="text-sm font-medium">{element.label}</div>
                          <div className="text-xs text-gray-500">SAR {element.price.toLocaleString()}</div>
                        </div>
                        <Package className="w-4 h-4 text-gray-400" />
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Pricing Summary */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mt-4">
              <h4 className="font-medium text-blue-800 mb-2">Pricing Summary</h4>
              <div className="text-sm text-blue-700 space-y-1">
                <div className="flex justify-between">
                  <span>Elements:</span>
                  <span>SAR {design.pricing.elements.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Materials:</span>
                  <span>SAR {design.pricing.materials.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Lighting:</span>
                  <span>SAR {design.pricing.lighting.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Setup:</span>
                  <span>SAR {design.pricing.setup.toLocaleString()}</span>
                </div>
                <div className="flex justify-between font-medium border-t border-blue-300 pt-1">
                  <span>Total:</span>
                  <span>SAR {design.pricing.total.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'style' && (
          <div className="space-y-4">
            {/* Colors */}
            <div>
              <h4 className="font-medium text-gray-800 mb-2">Colors</h4>
              <div className="space-y-2">
                {Object.entries(design.colors).map(([key, value]) => (
                  <div key={key} className="flex items-center space-x-2">
                    <label className="w-16 text-sm capitalize">{key}:</label>
                    <input
                      type="color"
                      value={value}
                      onChange={(e) => handleColorChange(key as any, e.target.value)}
                      className="w-8 h-8 rounded border border-gray-300"
                    />
                    <span className="text-xs text-gray-500">{value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Materials */}
            <div>
              <h4 className="font-medium text-gray-800 mb-2">Materials</h4>
              <div className="space-y-2">
                {Object.entries(design.materials).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-sm capitalize mb-1">{key}:</label>
                    <select
                      value={value}
                      onChange={(e) => handleMaterialChange(key as any, e.target.value)}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                    >
                      {key === 'floor' && (
                        <>
                          <option value="wood">Wood</option>
                          <option value="marble">Marble</option>
                          <option value="concrete">Concrete</option>
                          <option value="carpet">Carpet</option>
                        </>
                      )}
                      {key === 'walls' && (
                        <>
                          <option value="drywall">Drywall</option>
                          <option value="glass">Glass</option>
                          <option value="metal">Metal</option>
                          <option value="fabric">Fabric</option>
                          <option value="led_panel">LED Panel</option>
                        </>
                      )}
                      {key === 'ceiling' && (
                        <>
                          <option value="standard">Standard</option>
                          <option value="suspended">Suspended</option>
                          <option value="exposed">Exposed</option>
                          <option value="curved">Curved</option>
                          <option value="led_sky">LED Sky</option>
                        </>
                      )}
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {/* Style Presets */}
            <div>
              <h4 className="font-medium text-gray-800 mb-2">Style Presets</h4>
              <div className="grid grid-cols-2 gap-2">
                {['modern', 'futuristic', 'minimal', 'luxury', 'industrial', 'eco'].map((style) => (
                  <button
                    key={style}
                    onClick={() => onDesignUpdate({ style: style as any })}
                    className={`px-3 py-2 rounded text-sm capitalize transition-colors ${
                      design.style === style
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {style}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'dimensions' && (
          <div className="space-y-4">
            <h4 className="font-medium text-gray-800 mb-2">Booth Dimensions</h4>
            {Object.entries(design.dimensions).map(([key, value]) => (
              <div key={key}>
                <label className="block text-sm capitalize mb-1">{key} (meters):</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="range"
                    min={key === 'height' ? 2 : 3}
                    max={key === 'height' ? 5 : 20}
                    step={0.5}
                    value={value}
                    onChange={(e) => handleDimensionChange(key as any, parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <input
                    type="number"
                    min={key === 'height' ? 2 : 3}
                    max={key === 'height' ? 5 : 20}
                    step={0.5}
                    value={value}
                    onChange={(e) => handleDimensionChange(key as any, parseFloat(e.target.value))}
                    className="w-16 px-2 py-1 border border-gray-300 rounded text-sm"
                  />
                </div>
              </div>
            ))}
            
            <div className="bg-gray-50 rounded-lg p-3 mt-4">
              <h5 className="font-medium text-gray-800 mb-1">Total Area</h5>
              <p className="text-sm text-gray-600">
                {(design.dimensions.width * design.dimensions.depth).toFixed(1)} m²
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DesignControls;