
import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'glass';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  loading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  icon, 
  loading,
  className = '',
  disabled,
  ...props 
}) => {
  
  const baseStyles = "inline-flex items-center justify-center font-medium transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed rounded uppercase tracking-wider relative overflow-hidden group";
  
  const variants = {
    primary: "bg-cyber-accent/10 text-cyber-accent border border-cyber-accent/50 hover:bg-cyber-accent hover:text-cyber-950 hover:shadow-[0_0_20px_rgba(100,255,218,0.4)]",
    secondary: "bg-cyber-800 text-gray-300 border border-cyber-700 hover:border-gray-500 hover:text-white hover:bg-cyber-700",
    danger: "bg-red-500/10 text-red-500 border border-red-500/50 hover:bg-red-500 hover:text-white hover:shadow-[0_0_20px_rgba(239,68,68,0.4)]",
    ghost: "bg-transparent text-gray-400 hover:text-white hover:bg-white/5",
    glass: "glass-panel-light text-white hover:bg-white/10 hover:border-white/20"
  };

  const sizes = {
    xs: "text-[10px] px-2 py-1 gap-1",
    sm: "text-xs px-3 py-1.5 gap-1.5",
    md: "text-xs px-5 py-2.5 gap-2",
    lg: "text-sm px-8 py-3 gap-3",
  };

  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {/* Button sheen effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 pointer-events-none"></div>
      
      {loading && (
        <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
      )}
      {!loading && icon}
      <span className="relative z-10">{children}</span>
    </button>
  );
};
