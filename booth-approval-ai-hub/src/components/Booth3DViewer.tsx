import React, { Suspense, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Environment, Html, Sparkles } from '@react-three/drei';
import * as THREE from 'three';

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

interface Booth3DViewerProps {
  design: BoothDesign;
  selectedElement: string | null;
  onElementSelect: (id: string | null) => void;
  onElementPositionChange?: (elementId: string, newPosition: [number, number, number]) => void;
  className?: string;
}

// Enhanced Material Properties
const MATERIAL_PROPERTIES = {
  wood: { color: '#8B4513', roughness: 0.8, metalness: 0.1, emissive: '#000000', emissiveIntensity: 0 },
  marble: { color: '#F8F8FF', roughness: 0.05, metalness: 0.1, emissive: '#000000', emissiveIntensity: 0 },
  concrete: { color: '#696969', roughness: 0.95, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  carpet: { color: '#8B0000', roughness: 1.0, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  metal: { color: '#C0C0C0', roughness: 0.1, metalness: 0.95, emissive: '#000000', emissiveIntensity: 0 },
  glass: { color: '#87CEEB', roughness: 0.0, metalness: 0.0, transparent: true, opacity: 0.3, emissive: '#000000', emissiveIntensity: 0 },
  drywall: { color: '#F5F5F5', roughness: 0.8, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  fabric: { color: '#4B0082', roughness: 0.9, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  led_panel: { color: '#00BFFF', roughness: 0.1, metalness: 0.3, emissive: '#00BFFF', emissiveIntensity: 0.5 },
  bamboo: { color: '#DAA520', roughness: 0.7, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  standard: { color: '#FFFFFF', roughness: 0.6, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  suspended: { color: '#E0E0E0', roughness: 0.4, metalness: 0.2, emissive: '#000000', emissiveIntensity: 0 },
  exposed: { color: '#654321', roughness: 0.8, metalness: 0.1, emissive: '#000000', emissiveIntensity: 0 },
  curved: { color: '#FFF8DC', roughness: 0.3, metalness: 0.0, emissive: '#000000', emissiveIntensity: 0 },
  led_sky: { color: '#87CEEB', roughness: 0.1, metalness: 0.0, emissive: '#87CEEB', emissiveIntensity: 0.8 }
};

// Animated Hologram Component
const AnimatedHologram: React.FC<{ element: any }> = ({ element }) => {
  const mesh = React.useRef<THREE.Mesh>(null);
  const [time, setTime] = React.useState(0);
  
  useFrame((state) => {
    if (mesh.current) {
      const t = state.clock.getElapsedTime();
      setTime(t);
      // Floating animation
      mesh.current.position.y = Math.sin(t * 2) * 0.1 + element.size[1] * 0.3;
      // Rotation animation
      mesh.current.rotation.y = t * 0.5;
    }
  });

  return (
    <group>
      {/* Holographic projection */}
      <mesh ref={mesh}>
        <coneGeometry args={[element.size[0] * 0.8, element.size[1] * 0.4, 8]} />
        <meshStandardMaterial 
          color="#00ffff" 
          transparent={true} 
          opacity={0.6 + Math.sin(time * 3) * 0.2}
          emissive="#00ffff"
          emissiveIntensity={0.3 + Math.sin(time * 2) * 0.1}
        />
      </mesh>
      
      {/* Sparkle effects */}
      <Sparkles 
        count={50} 
        scale={[element.size[0] * 2, element.size[1], element.size[2] * 2]} 
        size={2} 
        speed={0.5}
        color="#00ffff"
      />
      
      {/* Hologram content */}
      <Text
        position={[0, element.size[1] * 0.3, 0]}
        fontSize={0.2}
        color="#00ffff"
        anchorX="center"
        anchorY="middle"
        rotation={[0, time * 0.5, 0]}
      >
        HOLOGRAM
      </Text>
    </group>
  );
};

// Video Wall Component
const VideoWall: React.FC<{ element: any }> = ({ element }) => {
  const [time, setTime] = React.useState(0);
  
  useFrame((state) => {
    setTime(state.clock.getElapsedTime());
  });

  return (
    <group>
      {/* Screen bezel */}
      <mesh>
        <boxGeometry args={element.size as [number, number, number]} />
        <meshStandardMaterial 
          color="#1a1a1a" 
          metalness={0.8} 
          roughness={0.2}
        />
      </mesh>
      
      {/* Active screen */}
      <mesh position={[0, 0, element.size[2] / 2 + 0.01]}>
        <planeGeometry args={[element.size[0] * 0.9, element.size[1] * 0.9]} />
        <meshStandardMaterial 
          color={`hsl(${(time * 50) % 360}, 70%, 50%)`}
          emissive={`hsl(${(time * 50) % 360}, 70%, 20%)`}
          emissiveIntensity={0.4}
          transparent={true}
          opacity={0.9}
        />
      </mesh>
      
      {/* Video content overlay */}
      <Html
        position={[0, 0, element.size[2] / 2 + 0.02]}
        style={{
          width: `${element.size[0] * 90}px`,
          height: `${element.size[1] * 90}px`,
          background: 'linear-gradient(45deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: '12px',
          fontWeight: 'bold'
        }}
      >
        <div style={{ textAlign: 'center' }}>
          📺 LIVE VIDEO<br/>
          <div style={{ fontSize: '10px', opacity: 0.8 }}>
            Product Demo
          </div>
        </div>
      </Html>
      
      {/* Screen glow */}
      <pointLight 
        position={[0, 0, element.size[2] / 2 + 0.5]} 
        color={`hsl(${(time * 50) % 360}, 70%, 50%)`}
        intensity={0.5} 
        distance={3}
      />
    </group>
  );
};

// Enhanced Element Component
const EnhancedElement: React.FC<{
  element: any;
  isSelected: boolean;
  onSelect: (e: any) => void;
  getMaterialProps: (material: string) => any;
  onElementPositionChange?: (elementId: string, newPosition: [number, number, number]) => void;
}> = ({ element, isSelected, onSelect, getMaterialProps, onElementPositionChange }) => {
  const meshRef = useRef<THREE.Group>(null);
  
  // Setup drag functionality with mouse events
  const handleMouseDown = (e: any) => {
    e.stopPropagation();
    if (!isSelected || !onElementPositionChange) return;
    
    // Simple drag implementation - in a real app you might use a more sophisticated drag library
    const startPosition = { x: e.clientX, y: e.clientY };
    const startElementPosition = [...element.position] as [number, number, number];
    
    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = (moveEvent.clientX - startPosition.x) * 0.01;
      const deltaZ = (moveEvent.clientY - startPosition.y) * 0.01;
      
      // Constrain to booth boundaries
      const boothWidth = 6; // Default booth width
      const boothDepth = 6; // Default booth depth
      
      const newX = Math.max(-boothWidth/2 + element.size[0]/2, 
                          Math.min(boothWidth/2 - element.size[0]/2, 
                                  startElementPosition[0] + deltaX));
      const newZ = Math.max(-boothDepth/2 + element.size[2]/2, 
                          Math.min(boothDepth/2 - element.size[2]/2, 
                                  startElementPosition[2] + deltaZ));
      
      const newPosition: [number, number, number] = [newX, element.position[1], newZ];
      onElementPositionChange(element.id, newPosition);
    };
    
    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  
  const renderElementGeometry = () => {
    switch (element.type) {
      case 'hologram_projector':
        return (
          <group>
            <mesh>
              <cylinderGeometry args={[element.size[0] / 2, element.size[0] / 3, element.size[1] * 0.6, 16]} />
              <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.2} />
            </mesh>
            <AnimatedHologram element={element} />
          </group>
        );
      
      case 'video_wall':
      case 'curved_display':
      case 'screen':
        return <VideoWall element={element} />;
      
      case 'interactive_kiosk':
        return (
          <group>
            {/* Base Pedestal */}
            <mesh position={[0, element.size[1] * 0.05, 0]} castShadow receiveShadow>
              <cylinderGeometry args={[element.size[0] * 0.4, element.size[0] * 0.5, element.size[1] * 0.1, 16]} />
              <meshStandardMaterial color="#1a202c" metalness={0.8} roughness={0.2} />
            </mesh>
            {/* Kiosk Stand */}
            <mesh position={[0, element.size[1] * 0.4, 0]} castShadow receiveShadow>
              <cylinderGeometry args={[element.size[0] * 0.15, element.size[0] * 0.15, element.size[1] * 0.7, 12]} />
              <meshStandardMaterial color="#2d3748" metalness={0.7} roughness={0.3} />
            </mesh>
            {/* Screen Housing */}
            <mesh position={[0, element.size[1] * 0.8, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 0.9, element.size[1] * 0.4, 0.1]} />
              <meshStandardMaterial color="#1a1a1a" metalness={0.9} roughness={0.1} />
            </mesh>
            {/* Active Screen */}
            <mesh position={[0, element.size[1] * 0.8, 0.06]} castShadow>
              <planeGeometry args={[element.size[0] * 0.8, element.size[1] * 0.35]} />
              <meshStandardMaterial 
                color="#0066cc" 
                emissive="#004499" 
                emissiveIntensity={0.5}
                toneMapped={false}
              />
            </mesh>
            {/* Control Panel */}
            <mesh position={[0, element.size[1] * 0.5, element.size[2] * 0.4]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 0.6, 0.1, element.size[2] * 0.3]} />
              <meshStandardMaterial color="#374151" metalness={0.5} roughness={0.7} />
            </mesh>

          </group>
        );
      
      case 'seating':
        return (
          <group>
            {/* Legs */}
            {[[-element.size[0]*0.35, 0, -element.size[2]*0.35], 
              [element.size[0]*0.35, 0, -element.size[2]*0.35],
              [-element.size[0]*0.35, 0, element.size[2]*0.35], 
              [element.size[0]*0.35, 0, element.size[2]*0.35]].map((pos, i) => (
              <mesh key={i} position={[pos[0], element.size[1] * 0.2, pos[2]]} castShadow receiveShadow>
                <cylinderGeometry args={[0.05, 0.05, element.size[1] * 0.4, 8]} />
                <meshStandardMaterial color="#2c1810" metalness={0.1} roughness={0.9} />
              </mesh>
            ))}
            {/* Seat Cushion */}
            <mesh position={[0, element.size[1] * 0.4, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 0.9, element.size[1] * 0.15, element.size[2] * 0.85]} />
              <meshStandardMaterial color="#8B4513" roughness={0.8} />
            </mesh>
            {/* Seat Frame */}
            <mesh position={[0, element.size[1] * 0.32, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0], element.size[1] * 0.05, element.size[2] * 0.9]} />
              <meshStandardMaterial color="#2c1810" metalness={0.1} roughness={0.9} />
            </mesh>
            {/* Backrest Cushion */}
            <mesh position={[0, element.size[1] * 0.7, -element.size[2] * 0.4]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 0.9, element.size[1] * 0.5, element.size[2] * 0.1]} />
              <meshStandardMaterial color="#8B4513" roughness={0.8} />
            </mesh>
            {/* Backrest Frame */}
            <mesh position={[0, element.size[1] * 0.7, -element.size[2] * 0.45]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0], element.size[1] * 0.55, element.size[2] * 0.05]} />
              <meshStandardMaterial color="#2c1810" metalness={0.1} roughness={0.9} />
            </mesh>
            {/* Armrests */}
            <mesh position={[-element.size[0] * 0.45, element.size[1] * 0.55, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.1, 0.1, element.size[2] * 0.6]} />
              <meshStandardMaterial color="#2c1810" metalness={0.1} roughness={0.9} />
            </mesh>
            <mesh position={[element.size[0] * 0.45, element.size[1] * 0.55, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.1, 0.1, element.size[2] * 0.6]} />
              <meshStandardMaterial color="#2c1810" metalness={0.1} roughness={0.9} />
            </mesh>
          </group>
        );
      
      case 'reception_counter':
        return (
          <group>
            {/* Base Cabinet */}
            <mesh position={[0, element.size[1] * 0.4, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0], element.size[1] * 0.8, element.size[2]]} />
              <meshStandardMaterial color="#2d3748" metalness={0.2} roughness={0.7} />
            </mesh>
            {/* Counter Top */}
            <mesh position={[0, element.size[1] * 0.85, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 1.05, element.size[1] * 0.05, element.size[2] * 1.05]} />
              <meshStandardMaterial color="#f7fafc" metalness={0.1} roughness={0.2} />
            </mesh>
            {/* Front Panel */}
            <mesh position={[0, element.size[1] * 0.4, element.size[2] * 0.52]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 0.95, element.size[1] * 0.75, 0.02]} />
              <meshStandardMaterial color="#4a5568" metalness={0.3} roughness={0.6} />
            </mesh>
            {/* Logo Area */}
            <mesh position={[0, element.size[1] * 0.6, element.size[2] * 0.53]} castShadow>
              <boxGeometry args={[element.size[0] * 0.4, element.size[1] * 0.2, 0.01]} />
              <meshStandardMaterial color="#1a365d" emissive="#1a365d" emissiveIntensity={0.2} />
            </mesh>
            {/* Side Panels */}
            <mesh position={[-element.size[0] * 0.52, element.size[1] * 0.4, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.02, element.size[1] * 0.8, element.size[2]]} />
              <meshStandardMaterial color="#2d3748" metalness={0.2} roughness={0.7} />
            </mesh>
            <mesh position={[element.size[0] * 0.52, element.size[1] * 0.4, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.02, element.size[1] * 0.8, element.size[2]]} />
              <meshStandardMaterial color="#2d3748" metalness={0.2} roughness={0.7} />
            </mesh>
            {/* Monitor on Top */}
            <mesh position={[element.size[0] * 0.3, element.size[1] * 1.1, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.4, 0.3, 0.05]} />
              <meshStandardMaterial color="#000000" metalness={0.9} roughness={0.1} />
            </mesh>
            <mesh position={[element.size[0] * 0.3, element.size[1] * 1.1, 0.026]} castShadow>
              <planeGeometry args={[0.35, 0.25]} />
              <meshStandardMaterial color="#0066cc" emissive="#004499" emissiveIntensity={0.3} />
            </mesh>
          </group>
        );
      
      case 'banner_stand':
        return (
          <group>
            {/* Pole */}
            <mesh position={[0, element.size[1] / 2, 0]}>
              <cylinderGeometry args={[0.02, 0.02, element.size[1], 8]} />
              <meshStandardMaterial color="#C0C0C0" metalness={0.8} roughness={0.2} />
            </mesh>
            {/* Banner */}
            <mesh position={[0, element.size[1] * 0.7, 0]}>
              <planeGeometry args={[element.size[0], element.size[1] * 0.6]} />
              <meshStandardMaterial color="#ffffff" />
            </mesh>
          </group>
        );
      
      case 'coffee_bar':
        return (
          <group>
            {/* Base Counter */}
            <mesh position={[0, element.size[1] * 0.45, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0], element.size[1] * 0.9, element.size[2]]} />
              <meshStandardMaterial color="#654321" roughness={0.8} />
            </mesh>
            {/* Counter Top */}
            <mesh position={[0, element.size[1] * 0.93, 0]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 1.02, element.size[1] * 0.06, element.size[2] * 1.02]} />
              <meshStandardMaterial color="#8B4513" roughness={0.6} />
            </mesh>
            {/* Coffee Machine */}
            <mesh position={[element.size[0] * 0.25, element.size[1] * 1.15, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.35, 0.4, 0.35]} />
              <meshStandardMaterial color="#2d3748" metalness={0.8} roughness={0.2} />
            </mesh>
            {/* Machine Display */}
            <mesh position={[element.size[0] * 0.25, element.size[1] * 1.25, 0.18]} castShadow>
              <boxGeometry args={[0.15, 0.1, 0.01]} />
              <meshStandardMaterial color="#000000" emissive="#004499" emissiveIntensity={0.4} />
            </mesh>
            {/* Cups Stack */}
            <mesh position={[-element.size[0] * 0.25, element.size[1] * 1.02, 0.2]} castShadow receiveShadow>
              <cylinderGeometry args={[0.08, 0.08, 0.15, 8]} />
              <meshStandardMaterial color="#ffffff" roughness={0.3} />
            </mesh>
            {/* Sugar/Stirrer Container */}
            <mesh position={[-element.size[0] * 0.1, element.size[1] * 1.02, 0.3]} castShadow receiveShadow>
              <boxGeometry args={[0.1, 0.08, 0.1]} />
              <meshStandardMaterial color="#8B4513" roughness={0.7} />
            </mesh>
            {/* Menu Board */}
            <mesh position={[0, element.size[1] * 1.5, -element.size[2] * 0.52]} castShadow receiveShadow>
              <boxGeometry args={[element.size[0] * 0.8, 0.6, 0.02]} />
              <meshStandardMaterial color="#1a1a1a" roughness={0.9} />
            </mesh>
            <mesh position={[0, element.size[1] * 1.5, -element.size[2] * 0.51]} castShadow>
              <planeGeometry args={[element.size[0] * 0.75, 0.55]} />
              <meshStandardMaterial color="#ffd700" emissive="#cc9900" emissiveIntensity={0.1} />
            </mesh>
          </group>
        );
      
      default:
        return (
          <mesh>
            <boxGeometry args={element.size as [number, number, number]} />
            <meshStandardMaterial 
              {...getMaterialProps(element.material || 'wood')}
              color={element.color}
            />
          </mesh>
        );
    }
  };

  return (
    <group>
      <group
        ref={meshRef}
        position={element.position as [number, number, number]}
        rotation={element.rotation as [number, number, number] || [0, 0, 0]}
        onClick={onSelect}
        onPointerDown={handleMouseDown}
      >
        {renderElementGeometry()}
        
        {/* Selection highlight */}
        {isSelected && (
          <mesh>
            <boxGeometry args={[
              element.size[0] * 1.1, 
              element.size[1] * 1.1, 
              element.size[2] * 1.1
            ]} />
            <meshStandardMaterial 
              color="#ffff00" 
              transparent={true} 
              opacity={0.2}
              wireframe={true}
            />
          </mesh>
        )}
      </group>
      
      {/* Element label when selected */}
      {isSelected && (
        <Text
          position={[
            element.position[0], 
            element.position[1] + element.size[1] / 2 + 0.3, 
            element.position[2]
          ]}
          fontSize={0.2}
          color="#fbbf24"
          anchorX="center"
          anchorY="middle"
        >
          ✨ {element.label || element.type}
        </Text>
      )}
    </group>
  );
};

// Main Booth 3D Component
const Booth3D: React.FC<{ 
  design: BoothDesign; 
  selectedElement: string | null; 
  onElementSelect: (id: string | null) => void;
  onElementPositionChange?: (elementId: string, newPosition: [number, number, number]) => void;
}> = ({ design, selectedElement, onElementSelect, onElementPositionChange }) => {
  
  const safeDesign = React.useMemo(() => ({
    dimensions: {
      width: Number(design?.dimensions?.width) || 6,
      depth: Number(design?.dimensions?.depth) || 6,
      height: Number(design?.dimensions?.height) || 3
    },
    materials: design?.materials || { floor: 'wood', walls: 'drywall', ceiling: 'standard' },
    colors: {
      primary: design?.colors?.primary || '#3B82F6',
      secondary: design?.colors?.secondary || '#F3F4F6',
      accent: design?.colors?.accent || '#10B981'
    },
    branding: design?.branding || { companyName: 'Your Company', brandColors: [] },
    elements: Array.isArray(design?.elements) ? design.elements : [],
    lighting: {
      ambient: Number(design?.lighting?.ambient) || 0.4,
      spotlights: Array.isArray(design?.lighting?.spotlights) ? design.lighting.spotlights : []
    },
    style: design?.style || 'modern'
  }), [design]);

  const getStyleBasedColors = (style: string) => {
    const styleColors = {
      modern: { primary: '#3B82F6', secondary: '#F8FAFC', accent: '#10B981' },
      futuristic: { primary: '#8B5CF6', secondary: '#1E1B4B', accent: '#06B6D4' },
      minimal: { primary: '#6B7280', secondary: '#F9FAFB', accent: '#F59E0B' },
      luxury: { primary: '#7C3AED', secondary: '#F8FAFC', accent: '#F59E0B' },
      industrial: { primary: '#374151', secondary: '#E5E7EB', accent: '#EF4444' },
      eco: { primary: '#10B981', secondary: '#F3F4F6', accent: '#84CC16' }
    };
    return styleColors[style as keyof typeof styleColors] || styleColors.modern;
  };

  const getMaterialProps = (materialType: string) => {
    const props = MATERIAL_PROPERTIES[materialType as keyof typeof MATERIAL_PROPERTIES] || MATERIAL_PROPERTIES.wood;
    const styleColors = getStyleBasedColors(safeDesign.style);
    
    // Modify material properties based on style
    let modifiedProps = { ...props };
    
    if (safeDesign.style === 'futuristic') {
      modifiedProps.metalness = Math.min(1.0, props.metalness + 0.3);
      modifiedProps.emissiveIntensity = props.emissiveIntensity + 0.2;
    } else if (safeDesign.style === 'luxury') {
      modifiedProps.roughness = Math.max(0.0, props.roughness - 0.2);
      modifiedProps.metalness = Math.min(1.0, props.metalness + 0.2);
    } else if (safeDesign.style === 'minimal') {
      modifiedProps.roughness = Math.min(1.0, props.roughness + 0.2);
      modifiedProps.metalness = Math.max(0.0, props.metalness - 0.2);
    } else if (safeDesign.style === 'industrial') {
      modifiedProps.roughness = Math.min(1.0, props.roughness + 0.3);
      modifiedProps.metalness = Math.min(1.0, props.metalness + 0.4);
    } else if (safeDesign.style === 'eco') {
      modifiedProps.roughness = Math.min(1.0, props.roughness + 0.1);
    }
    
    return {
      color: modifiedProps.color,
      roughness: modifiedProps.roughness,
      metalness: modifiedProps.metalness,
      transparent: (modifiedProps as any).transparent || false,
      opacity: (modifiedProps as any).opacity || 1,
      emissive: modifiedProps.emissive || '#000000',
      emissiveIntensity: modifiedProps.emissiveIntensity || 0
    };
  };

  return (
    <group>
      {/* Floor */}
      <mesh 
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        onClick={() => onElementSelect(null)}
        receiveShadow
      >
        <planeGeometry args={[safeDesign.dimensions.width, safeDesign.dimensions.depth]} />
        <meshStandardMaterial 
          {...getMaterialProps(safeDesign.materials.floor)}
          color={getStyleBasedColors(safeDesign.style).secondary}
        />
      </mesh>

      {/* Grid helper */}
      <gridHelper 
        args={[Math.max(safeDesign.dimensions.width, safeDesign.dimensions.depth), 20]} 
        position={[0, 0.01, 0]}
      />

      {/* Back Wall */}
      <mesh position={[0, safeDesign.dimensions.height / 2, -safeDesign.dimensions.depth / 2]} onClick={() => onElementSelect(null)} castShadow receiveShadow>
        <boxGeometry args={[safeDesign.dimensions.width, safeDesign.dimensions.height, 0.1]} />
        <meshStandardMaterial 
          {...getMaterialProps(safeDesign.materials.walls)}
          color={getStyleBasedColors(safeDesign.style).primary}
        />
      </mesh>

      {/* Left Wall */}
      <mesh position={[-safeDesign.dimensions.width / 2, safeDesign.dimensions.height / 2, 0]} onClick={() => onElementSelect(null)} castShadow receiveShadow>
        <boxGeometry args={[0.1, safeDesign.dimensions.height, safeDesign.dimensions.depth]} />
        <meshStandardMaterial 
          {...getMaterialProps(safeDesign.materials.walls)}
          color={getStyleBasedColors(safeDesign.style).primary}
        />
      </mesh>

      {/* Right Wall */}
      <mesh position={[safeDesign.dimensions.width / 2, safeDesign.dimensions.height / 2, 0]} onClick={() => onElementSelect(null)} castShadow receiveShadow>
        <boxGeometry args={[0.1, safeDesign.dimensions.height, safeDesign.dimensions.depth]} />
        <meshStandardMaterial 
          {...getMaterialProps(safeDesign.materials.walls)}
          color={getStyleBasedColors(safeDesign.style).primary}
        />
      </mesh>

      {/* Ceiling/Roof */}
      <mesh position={[0, safeDesign.dimensions.height, 0]} onClick={() => onElementSelect(null)} castShadow receiveShadow>
        <boxGeometry args={[safeDesign.dimensions.width, 0.1, safeDesign.dimensions.depth]} />
        <meshStandardMaterial {...getMaterialProps(safeDesign.materials.ceiling)} />
      </mesh>

      {/* Company Name */}
      {safeDesign.branding.companyName && (
        <Text
          position={[0, safeDesign.dimensions.height * 0.7, -safeDesign.dimensions.depth / 2 + 0.1]}
          fontSize={0.5}
          color={safeDesign.colors.primary}
          anchorX="center"
          anchorY="middle"
        >
          {safeDesign.branding.companyName}
        </Text>
      )}

      {/* Design Elements */}
      {safeDesign.elements.map((element) => {
        if (!element?.id) return null;
        
        const isSelected = selectedElement === element.id;
        
        return (
          <EnhancedElement
            key={element.id}
            element={element}
            isSelected={isSelected}
            onSelect={(e) => {
              e.stopPropagation();
              onElementSelect(element.id);
            }}
            getMaterialProps={getMaterialProps}
            onElementPositionChange={onElementPositionChange}
          />
        );
      })}

      {/* Lighting */}
      <ambientLight intensity={safeDesign.lighting.ambient * 1.2} color="#ffffff" />
      <directionalLight
        position={[10, 10, 5]}
        intensity={1.5}
        color="#ffffff"
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
    </group>
  );
};

// Error Boundary Component
class Canvas3DErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('3D Canvas Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full bg-gray-100 rounded-lg">
          <div className="text-center p-8">
            <h3 className="text-lg font-semibold mb-2">3D Viewer Unavailable</h3>
            <p className="text-gray-600 mb-4">
              The 3D viewer couldn't load. This might be due to WebGL support issues.
            </p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Loading Component
const Canvas3DLoader: React.FC = () => (
  <div className="flex items-center justify-center h-full bg-gradient-to-br from-blue-900 via-purple-900 to-pink-900">
    <div className="text-center text-white">
      <div className="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
      <p className="text-lg font-semibold">🚀 Loading 3D Environment</p>
    </div>
  </div>
);

// Main Component
const Booth3DViewer: React.FC<Booth3DViewerProps> = ({
  design,
  selectedElement,
  onElementSelect,
  onElementPositionChange,
  className = ""
}) => {
  return (
    <div className={`relative ${className}`}>
      <Canvas3DErrorBoundary>
        <Suspense fallback={<Canvas3DLoader />}>
          <Canvas
            camera={{ 
              position: [8, 6, 8], 
              fov: 50,
              near: 0.1,
              far: 1000
            }}
            shadows
            onCreated={({ gl, scene }) => {
              try {
                gl.outputColorSpace = THREE.SRGBColorSpace;
                scene.background = new THREE.Color('#1e293b');
              } catch (error) {
                console.error('Canvas setup error:', error);
              }
            }}
            onError={(error) => {
              console.error('Canvas error:', error);
            }}
          >
            <Environment preset="warehouse" />
            
            {/* Exhibition Hall Lighting */}
            <ambientLight intensity={0.3} color="#f0f0f0" />
            <directionalLight
              position={[10, 15, 10]}
              intensity={1.2}
              color="#ffffff"
              castShadow
              shadow-mapSize={[2048, 2048]}
              shadow-camera-far={50}
              shadow-camera-left={-10}
              shadow-camera-right={10}
              shadow-camera-top={10}
              shadow-camera-bottom={-10}
            />
            
            {/* Exhibition Hall Spotlights */}
            <spotLight
              position={[5, 8, 5]}
              intensity={0.8}
              angle={Math.PI / 6}
              penumbra={0.2}
              color="#ffffff"
              castShadow
            />
            <spotLight
              position={[-5, 8, 5]}
              intensity={0.8}
              angle={Math.PI / 6}
              penumbra={0.2}
              color="#ffffff"
              castShadow
            />
            
            {/* Ambient exhibition atmosphere */}
            <fog attach="fog" args={['#f0f0f0', 20, 100]} />
            
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              maxPolarAngle={Math.PI * 0.75}
              minDistance={3}
              maxDistance={50}
              target={[0, 1, 0]}
            />
            
            <Booth3D
              design={design}
              selectedElement={selectedElement}
              onElementSelect={onElementSelect}
              onElementPositionChange={onElementPositionChange}
            />
          </Canvas>
        </Suspense>
      </Canvas3DErrorBoundary>
    </div>
  );
};

export default Booth3DViewer;