
import React from 'react';
import { StudioMode } from '../types';
import { Network, Database, Grid, Activity, Box, GitMerge, Compass } from 'lucide-react';

interface StudioNavProps {
  currentMode: StudioMode;
  onModeChange: (mode: StudioMode) => void;
}

export const StudioNav: React.FC<StudioNavProps> = ({ currentMode, onModeChange }) => {
  
  const groups = [
    {
      items: [
        { mode: StudioMode.GENERAL_ML, label: 'General ML', icon: <Compass size={20} /> },
        { mode: StudioMode.LOSS_LANDSCAPE, label: 'Loss Landscape', icon: <Activity size={20} /> },
      ]
    },
    {
      items: [
        { mode: StudioMode.NEURAL_ARCH, label: 'Neural Networks', icon: <Network size={20} /> },
        { mode: StudioMode.EMBEDDING_SPACE, label: 'Embeddings', icon: <Grid size={20} /> },
        { mode: StudioMode.RL_ARENA, label: 'RL Arena', icon: <Box size={20} /> },
      ]
    },
    {
      items: [
        { mode: StudioMode.CLUSTERING, label: 'Clustering', icon: <Database size={20} /> },
        { mode: StudioMode.DECISION_BOUNDARY, label: 'SVM / Decision', icon: <GitMerge size={20} /> },
      ]
    }
  ];

  return (
    <nav className="flex flex-col items-center py-6 gap-6 w-full">
      {groups.map((group, idx) => (
        <div key={idx} className="flex flex-col gap-4 w-full items-center">
          {/* Subtle Separator between groups */}
          {idx > 0 && <div className="w-8 h-[1px] bg-cyber-800/50"></div>}
          
          {group.items.map((item) => {
            const isActive = currentMode === item.mode;
            return (
              <button
                key={item.mode}
                onClick={() => onModeChange(item.mode)}
                className={`
                  relative group flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-300
                  ${isActive 
                    ? 'bg-cyber-accent text-cyber-900 shadow-[0_0_15px_rgba(100,255,218,0.4)] scale-110' 
                    : 'text-gray-400 hover:text-white hover:bg-white/10'}
                `}
              >
                {item.icon}
                
                {/* Tooltip */}
                <div className="absolute left-full top-1/2 -translate-y-1/2 ml-4 px-3 py-1.5 bg-cyber-900/95 border border-cyber-700 rounded-md shadow-xl opacity-0 group-hover:opacity-100 transition-all duration-200 pointer-events-none whitespace-nowrap z-50 backdrop-blur-md translate-x-[-10px] group-hover:translate-x-0">
                   <div className="text-[10px] font-bold text-cyber-accent tracking-widest uppercase">{item.label}</div>
                   {/* Triangle arrow */}
                   <div className="absolute left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-cyber-900 border-l border-b border-cyber-700 transform rotate-45"></div>
                </div>
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
};
