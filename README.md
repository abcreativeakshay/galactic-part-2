# Dracarys: Enterprise-Grade 3D ML Visualization Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Three.js](https://img.shields.io/badge/Three.js-3D%20Rendering-black)](https://threejs.org/)
[![Gemini API](https://img.shields.io/badge/Google-Gemini%20API-blue)](https://ai.google.dev/)

## 🌟 Overview

Dracarys is a cutting-edge, web-based 3D visualization platform designed for Machine Learning practitioners, researchers, and educators. By combining Three.js for high-fidelity 3D rendering with Google's Gemini API for AI-powered insights, Dracarys transforms complex ML concepts into interactive, visually stunning experiences.

**Experience ML like never before** - from neural network architectures to clustering algorithms, all visualized in real-time 3D.

## ✨ Key Features

### 🏠 **Intelligent Model Discovery**
- **Categorized Model Library**: Browse pre-defined ML architectures (Generative AI, Computer Vision, Deep Learning)
- **AI-Powered Search**: Describe a model in natural language, and Dracarys generates the visualization configuration
- **Quick Start**: One-click loading of popular architectures (ResNet, GPT, VAE, etc.)

### 🎮 **Interactive 3D Visualization Studio**
- **Real-time 3D Rendering**: Powered by Three.js with WebGL acceleration
- **Multiple Visualization Modes**:
  - Neural Architecture (nodes & connections)
  - Clustering algorithms
  - Decision boundaries
  - Loss landscapes
- **Full Scene Control**: Rotate, pan, and zoom with intuitive mouse controls
- **Animated Learning**: Watch "signal" particles flow through neural networks

### ⚙️ **Comprehensive Configuration**
- **Model Parameter Tuning**: Modify layer architectures, hyperparameters, and training settings
- **Visual Customization**: Adjust glow effects, particle density, wireframe modes, and color schemes
- **Cyberpunk Aesthetic**: Immersive dark theme with neon accents and glassmorphism effects

### 🤖 **AI-Assisted Workflow**
- **Gemini-Powered Chat Assistant**: Get explanations, debug concepts, and ask questions
- **Algorithm Trace**: Step-by-step breakdown of ML algorithms
- **Code Generation**: Generate boilerplate code in PyTorch or TensorFlow
- **Contextual Insights**: AI understands your current visualization for relevant help

### 📊 **Analytics & Export**
- **Real-time Telemetry**: Live metrics dashboard with loss/accuracy graphs
- **Export Capabilities**:
  - High-resolution PNG snapshots
  - 3D model export (GLB/GLTF format)
  - Shareable configurations

## 🚀 Quick Start

### Prerequisites
- Modern web browser with WebGL 2.0 support
- Google Gemini API key ([Get one here](https://ai.google.dev/))
- Node.js 18+ (for local development)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dracarys.git
cd dracarys
```

2. **Install dependencies**
```bash
npm install
```

3. **Configure environment**
```bash
cp .env.example .env.local
# Edit .env.local and add your Gemini API key
```

4. **Start development server**
```bash
npm run dev
```

5. **Open your browser**
Navigate to `http://localhost:3000`

## 📖 User Guide

### Getting Started
1. **Accept Terms**: First-time users must acknowledge the Terms of Service
2. **Home Dashboard**: Browse model categories or use the AI search
3. **Select a Model**: Click any model card or search for a custom architecture
4. **Enter Studio**: The 3D visualization loads automatically

### Studio Interface Tour
- **Center**: 3D canvas with interactive visualization
- **Bottom Dock**: Access all control panels (Model, Visual, Code, Chat, Export)
- **Left Sidebar**: Algorithm trace and telemetry
- **Top Right**: Animation controls (Play/Pause/Reset)

### Using AI Features
1. **Search for Models**: Type natural language descriptions in the home search bar
2. **Ask the Assistant**: Use the chat panel for explanations and help
3. **Generate Code**: Switch to Code panel and select your framework
4. **Get Algorithm Steps**: View the generated trace in the sidebar

### Customizing Visualizations
1. **Model Panel**: Add/remove layers, adjust hyperparameters
2. **Visual Panel**: Toggle effects, adjust colors, modify particle systems
3. **Real-time Updates**: All changes reflect immediately in the 3D view

## 🏗️ Architecture

```
Dracarys/
├── src/
│   ├── components/     # React components
│   │   ├── Home/       # Landing page components
│   │   ├── Studio/     # 3D visualization components
│   │   └── UI/         # Reusable UI elements
│   ├── lib/
│   │   ├── threejs/    # Three.js scene management
│   │   ├── gemini/     # Gemini API integration
│   │   └── utils/      # Utility functions
│   ├── types/          # TypeScript definitions
│   └── styles/         # Global styles and themes
├── public/             # Static assets
└── config/             # Build configurations
```

### Tech Stack
- **Frontend**: React 18, TypeScript, Vite
- **3D Engine**: Three.js, @react-three/fiber
- **AI Integration**: Google Gemini API
- **Styling**: Tailwind CSS, Framer Motion
- **State Management**: Zustand
- **Build Tool**: Vite

## 🔧 Configuration

### Environment Variables
```env
VITE_GEMINI_API_KEY=your_gemini_api_key_here
VITE_APP_ENV=development
VITE_MAX_MODEL_SIZE=1000
```

### Model Configuration Format
```json
{
  "name": "ResNet-50",
  "type": "NEURAL_ARCH",
  "layers": [
    {
      "type": "Conv2D",
      "units": 64,
      "activation": "ReLU"
    }
  ],
  "hyperparameters": {
    "learningRate": 0.001,
    "dropout": 0.5
  }
}
```

## 📱 Supported Browsers

- Chrome 90+ (recommended)
- Firefox 88+
- Safari 15+
- Edge 90+

**Note**: WebGL 2.0 is required for optimal performance.

## 🎯 Performance Targets

- **Rendering**: 60 FPS on supported hardware
- **UI Response**: <100ms for all interactions
- **API Calls**: Async with loading indicators
- **Model Limits**: Simplified visualization for models >1B parameters

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Write TypeScript with strict type checking
- Follow the existing code style (ESLint/Prettier configured)
- Add tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Three.js](https://threejs.org/) for incredible 3D graphics capabilities
- [Google Gemini](https://ai.google.dev/) for AI/ML intelligence
- React Three Fiber community for excellent examples and support
- All contributors and users of Dracarys

## 📞 Support

- **Documentation**: [Full documentation available here](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/dracarys/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/dracarys/discussions)
- **Email**: support@dracarys-ml.com

## 🚧 Roadmap

### Phase 1 (Current)
- [x] Core 3D visualization engine
- [x] Basic model library
- [x] Gemini API integration
- [x] Export functionality

### Phase 2 (Next)
- [ ] Collaborative visualization sessions
- [ ] Plugin system for custom visualizers
- [ ] Dataset visualization tools
- [ ] Mobile-responsive design

### Phase 3 (Future)
- [ ] Real-time training visualization
- [ ] AR/VR support
- [ ] Cloud save and sharing
- [ ] Advanced analytics dashboard

---

**Dracarys** - Where Machine Learning Meets Visual Artistry. Ignite your understanding.

*"A mind needs books as a sword needs a whetstone." - And now, ML needs visualization.*
