Software Requirements Specification (SRS)
Project: Dracarys
Version: 1.0
Date: [Current Date]
1. Introduction
1.1 Purpose
This document provides a detailed specification for the Dracarys project. Dracarys is a web-based, enterprise-grade 3D visualization platform designed for a wide array of Machine Learning (ML) models. It leverages Three.js for high-fidelity 3D rendering and the Google Gemini API to provide AI-powered insights, configuration generation, and code synthesis.
1.2 Scope
The platform is an interactive, educational, and analytical tool for ML practitioners, data scientists, students, and researchers. Its scope covers:
Model Browsing: A comprehensive library of pre-defined ML model architectures.
AI-Powered Architecture Generation: Creation of model visualization configurations from natural language text prompts.
Interactive 3D Visualization: Real-time, three-dimensional rendering of various ML concepts, including neural network topologies, clustering algorithms, decision boundaries, and loss landscapes.
Model & Visual Customization: In-depth control panels for modifying both ML model parameters and visual aesthetics.
AI-Assisted Analysis: An integrated Gemini-powered chat assistant for explaining concepts, generating algorithm steps, and producing boilerplate code.
Exporting: The ability to export visualizations as static images or 3D models.
1.3 Definitions, Acronyms, and Abbreviations
AI: Artificial Intelligence
ML: Machine Learning
UI: User Interface
UX: User Experience
WebGL: Web Graphics Library
Three.js: A cross-browser JavaScript library/API used to create and display animated 3D computer graphics in a web browser.
Gemini API: Google's family of generative AI models.
SRS: Software Requirements Specification
GLB/GLTF: GL Transmission Format; a standard file format for 3D scenes and models.
NN: Neural Network
SVM: Support Vector Machine
PCA: Principal Component Analysis
LLM: Large Language Model
2. Overall Description
2.1 Product Perspective
Dracarys is a standalone, client-side web application that runs entirely in the user's browser. It is dependent on a modern browser with WebGL support for its rendering capabilities and requires a valid Google Gemini API key (provided via an environment variable process.env.API_KEY) for its intelligent features.
2.2 Product Functions
The major functions of the Dracarys platform are:
Home Dashboard: Serves as the main entry point, providing access to the model library and AI-powered search.
Model Library: A categorized, searchable collection of common ML models that can be loaded into the visualization studio.
3D Visualization Studio: The core interactive environment where users can view and manipulate the 3D representation of the selected ML model.
Configuration Panels: A set of context-aware UI panels for fine-tuning model architecture, hyperparameters, and visual settings.
AI Assistant: An integrated chat interface that provides contextual help and generates relevant artifacts (code, explanations).
Export Module: Enables users to save their work as PNG images or GLB 3D models.
2.3 User Characteristics
The intended users of this platform include:
Machine Learning Engineers & Data Scientists: For analyzing, debugging, and presenting model architectures.
Students & Educators: As an interactive learning tool to understand complex ML concepts visually.
Researchers: To explore novel architectures and visualize high-dimensional data.
AI Enthusiasts: For general exploration and learning.
Users are expected to have a basic understanding of Machine Learning concepts.
2.4 Constraints
The application must run in a modern web browser with WebGL enabled.
Performance is dependent on the client's GPU capabilities.
All AI features are contingent on a valid and accessible Google Gemini API key.
The application is client-side only; no user data or configurations are persisted on a server.
Visualizations of extremely large models (e.g., billions of parameters) are simplified for performance.
2.5 Assumptions and Dependencies
The process.env.API_KEY environment variable is correctly configured and available at runtime.
The user has a stable internet connection for initial asset loading and for making API calls to the Gemini service.
The client-side environment has sufficient memory and processing power for smooth 3D rendering.
3. Specific Requirements
3.1 Functional Requirements
FR-1.1: Boot Sequence: The application shall display an animated loading screen upon initial startup.
FR-1.2: Home View: The default view shall be the "HOME" dashboard, featuring a hero section, the Dracarys logo, and the model selection interface.
FR-1.3: Studio View: Upon selecting or generating a model, the view shall transition to the "STUDIO" interface, which contains the 3D canvas and all related controls.
FR-1.4: Terms of Service: The application shall provide a modal displaying the terms of service, which the user must acknowledge.
FR-1.5: Toast Notifications: The system shall display temporary, non-blocking toast notifications for events like loading models, successful exports, or errors.
FR-2.1: Model Categories: The Home View shall display ML models grouped into logical categories (e.g., Generative AI, Computer Vision, Deep Learning).
FR-2.2: Model Selection: Users shall be able to select a specific model from a category to load its preset configuration into the Studio View.
FR-2.3: AI-Powered Model Search: The Home View shall feature a search input. When a user enters a model name (e.g., "ResNet-50") and submits, the system shall:
FR-2.3.1: Call the Gemini API (generateConfigFromModelName) with the user's query.
FR-2.3.2: Receive a generated MLModelConfig JSON object.
FR-2.3.3: Load the received configuration and transition to the Studio View.
FR-3.1: 3D Scene Rendering: The system shall render a 3D scene using Three.js, including a grid, starfield background, and appropriate lighting.
FR-3.2: Visualization Modes: The system must support rendering for different StudioMode configurations:
FR-3.2.1: Neural Architecture (NEURAL_ARCH): Visualize layers as arrangements of nodes (spheres/cubes) connected by lines. Shall differentiate between layer types (Dense, Conv2D, etc.).
FR-3.2.2: Other Modes: The system has stubs and types for other modes like CLUSTERING, LOSS_LANDSCAPE, etc., which would be rendered by generating appropriate data points, surfaces, or structures. (Note: Current ThreeScene implementation focuses primarily on NEURAL_ARCH).
FR-3.3: Scene Interaction: Users shall be able to rotate (orbit), pan, and zoom the 3D camera using mouse controls.
FR-3.4: Hover Effects: Hovering the mouse over an interactive element in the scene (e.g., a neuron) shall trigger a visual highlight.
FR-3.5: Animation: The system shall support real-time animations, such as "signal" particles flowing along connections in a neural network.
FR-3.6: Animation Controls: A persistent control dock shall provide buttons to Play, Pause, and Reset the visualization's animation and state (e.g., epoch counter).
FR-4.1: Control Dock: A central dock at the bottom of the Studio View shall provide access to various tool panels.
FR-4.2: Model Configuration Panel: This panel shall allow users to modify the parameters of the current MLModelConfig.
FR-4.2.1: Layer Management: For neural networks, users can add new layers, remove existing layers, and modify parameters for each layer (e.g., type, neuron count).
FR-4.2.2: Hyperparameter Tuning: Users can adjust global parameters like Learning Rate and Dropout Rate via sliders.
FR-4.3: Visual Settings Panel: This panel shall allow users to modify VisualSettings.
FR-4.3.1: Adjust visual parameters like Glow Intensity and Particle Density using sliders.
FR-4.3.2: Toggle boolean settings like Wireframe Mode.
FR-5.1: Algorithm Trace: The Studio View shall display a "Trace" window.
FR-5.1.1: Upon loading a model, the system shall call the Gemini API (generateSteps) to generate a list of procedural steps explaining the algorithm.
FR-5.1.2: These steps shall be displayed in the Trace window.
FR-5.2: Code Generation: The "Code" panel shall generate source code based on the current MLModelConfig.
FR-5.2.1: The user can select between PyTorch and TensorFlow as the target framework.
FR-5.2.2: The system shall generate and display corresponding boilerplate model definition code.
FR-5.3: AI Chat Assistant: The Studio View shall include a "Chat" window.
FR-5.3.1: Users can send text-based messages to an AI assistant.
FR-5.3.2: The system shall call the Gemini Chat API (chatWithExpert), maintaining conversation history, and display the model's response.
FR-6.1: Telemetry: The Studio View shall display a "Telemetry" window with simulated real-time metrics.
FR-6.1.1: It shall display evolving line graphs for key metrics (e.g., Loss, Accuracy).
FR-6.1.2: It shall display derived metrics (e.g., Precision, Recall).
FR-6.2: Export Panel: An "Export" panel shall provide options to save artifacts.
FR-6.2.1: PNG Snapshot: Users can export the current canvas view as a PNG image file.
FR-6.2.2: 3D Model Export: Users can export the current 3D scene geometry as a binary GLTF (.glb) file.
3.2 Non-Functional Requirements
NFR-1: Performance
NFR-1.1: Rendering Frame Rate: The 3D visualization shall target a rendering speed of 60 frames per second (FPS) on supported hardware.
NFR-1.2: UI Responsiveness: All UI elements (buttons, sliders, panels) must respond to user input in under 100ms.
NFR-1.3: API Latency: Gemini API calls should be handled asynchronously with clear loading indicators to the user.
NFR-2: Usability
NFR-2.1: Intuitiveness: The UI should be intuitive, with clear labels, icons, and tooltips to guide the user.
NFR-2.2: Consistency: The visual design and interaction patterns shall be consistent across the Home and Studio views.
NFR-3: Reliability
NFR-3.1: Error Handling: The application must gracefully handle potential Gemini API errors (e.g., network issues, invalid requests) and display an informative error message to the user via a toast notification.
NFR-3.2: Stability: The application should not crash or freeze the browser during normal operation, even with complex scenes or long-running animations.
NFR-4: Aesthetics
NFR-4.1: Visual Theme: The application must maintain a consistent, high-fidelity "cyberpunk" aesthetic, characterized by dark backgrounds, neon accents (cyan, violet), and glassmorphism effects.
NFR-4.2: Typography: The application shall use the specified font families (Rajdhani, Inter, Fira Code) to maintain its visual identity.
NFR-4.3: Animations: All UI transitions and animations shall be smooth and adhere to the defined animation keyframes (e.g., fadeIn, slideUp).
