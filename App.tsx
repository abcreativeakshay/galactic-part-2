
import React, { useState, useEffect, useRef, Suspense } from 'react';
import { Button } from './components/Button';
import { analyzeModel, suggestOptimization, chatWithExpert, generateConfigFromModelName, generateSteps } from './services/geminiService';
import { MLModelConfig, StudioMode, LayerType, ActivationType, OptimizerType, ChatMessage, VizState, Toast, VisualSettings, AlgorithmStep } from './types';
import { Play, Pause, RefreshCw, MessageSquare, Zap, Cpu, Code, X, Check, ChevronRight, Eye, Sliders, AlertCircle, Info, Search, Layers, Plus, Trash2, Network, Activity, Box, GitMerge, Loader2, Database, Camera, ArrowLeft, Terminal, LayoutGrid, Monitor, Share2, Sparkles, ChevronDown, Minimize2, Maximize2, MoreHorizontal, Send, FileText, Shield, Download, Image, Wand2, Brain, Fingerprint, Mic, Component, ChevronLeft, BarChart3, Binary, Aperture } from 'lucide-react';
import { ThreeSceneHandle } from './components/ThreeScene';

const ThreeScene = React.lazy(() => import('./components/ThreeScene').then(module => ({ default: module.ThreeScene })));

// --- CONSTANTS & DATA ---

interface ModelDefinition {
    id: string;
    label: string;
    desc: string;
    mode: StudioMode; // The rendering mode this model uses
}

interface ModelCategory {
    id: string;
    title: string;
    icon: React.ReactNode;
    description: string;
    models: ModelDefinition[];
}

const MODEL_CATEGORIES: ModelCategory[] = [
    {
      id: 'GEN_AI',
      title: 'Generative AI & LLMs',
      icon: <Wand2 size={24} />,
      description: 'Diffusion models, GANs, Transformers, and LLMs.',
      models: [
        { id: 'stable-diffusion', label: 'Stable Diffusion', desc: 'Latent Diffusion Model', mode: StudioMode.GENERATIVE_AI },
        { id: 'llama', label: 'Llama 2/3', desc: 'Foundation LLM', mode: StudioMode.NLP },
        { id: 'gpt', label: 'GPT (Transformer)', desc: 'Generative Pre-trained Transformer', mode: StudioMode.NLP },
        { id: 'bert', label: 'BERT', desc: 'Bidirectional Encoder Representations', mode: StudioMode.NLP },
        { id: 't5', label: 'T5', desc: 'Text-to-Text Transfer Transformer', mode: StudioMode.NLP },
        { id: 'gan', label: 'GAN', desc: 'Generative Adversarial Network', mode: StudioMode.GENERATIVE_AI },
        { id: 'vae', label: 'VAE', desc: 'Variational Autoencoder', mode: StudioMode.GENERATIVE_AI },
        { id: 'diffusion', label: 'Diffusion Model', desc: 'Denoising Probabilistic Model', mode: StudioMode.GENERATIVE_AI }
      ]
    },
    {
        id: 'COMPUTER_VISION',
        title: 'Computer Vision',
        icon: <Aperture size={24} />,
        description: 'Object detection, segmentation, and classification.',
        models: [
            { id: 'cnn-standard', label: 'CNN (ResNet/VGG)', desc: 'Convolutional Neural Network', mode: StudioMode.COMPUTER_VISION },
            { id: 'yolo', label: 'YOLO v8/v9', desc: 'You Only Look Once', mode: StudioMode.COMPUTER_VISION },
            { id: 'vit', label: 'Vision Transformer', desc: 'Patch-based Attention', mode: StudioMode.COMPUTER_VISION },
            { id: 'clip', label: 'CLIP', desc: 'Contrastive Language-Image Pretraining', mode: StudioMode.COMPUTER_VISION },
            { id: 'sam', label: 'SAM', desc: 'Segment Anything Model', mode: StudioMode.COMPUTER_VISION },
            { id: 'detr', label: 'DETR', desc: 'Detection Transformer', mode: StudioMode.COMPUTER_VISION },
            { id: 'blip', label: 'BLIP', desc: 'Bootstrapping Language-Image Pre-training', mode: StudioMode.COMPUTER_VISION }
        ]
    },
    {
      id: 'DEEP_LEARNING',
      title: 'Deep Learning Core',
      icon: <Brain size={24} />,
      description: 'Fundamental neural architectures.',
      models: [
        { id: 'mlp', label: 'ANN / MLP', desc: 'Multi-Layer Perceptron', mode: StudioMode.NEURAL_ARCH },
        { id: 'rnn', label: 'RNN', desc: 'Recurrent Neural Network', mode: StudioMode.NEURAL_ARCH },
        { id: 'lstm', label: 'LSTM / GRU', desc: 'Long Short-Term Memory', mode: StudioMode.NEURAL_ARCH },
        { id: 'autoencoder', label: 'Autoencoder', desc: 'Compression & Reconstruction', mode: StudioMode.NEURAL_ARCH }
      ]
    },
    {
      id: 'CLASSICAL_ML',
      title: 'Supervised Learning',
      icon: <Binary size={24} />,
      description: 'Regression, Classification, and SVMs.',
      models: [
        { id: 'linear-reg', label: 'Linear Regression', desc: 'OLS Regression', mode: StudioMode.DECISION_BOUNDARY },
        { id: 'logistic-reg', label: 'Logistic Regression', desc: 'Binary Classification', mode: StudioMode.DECISION_BOUNDARY },
        { id: 'svm', label: 'SVM', desc: 'Support Vector Machine', mode: StudioMode.DECISION_BOUNDARY },
        { id: 'naive-bayes', label: 'Naive Bayes', desc: 'Probabilistic Classifier', mode: StudioMode.GENERAL_ML },
        { id: 'knn', label: 'KNN', desc: 'K-Nearest Neighbors', mode: StudioMode.GENERAL_ML },
        { id: 'decision-tree', label: 'Decision Tree', desc: 'Tree-based Classification', mode: StudioMode.GENERAL_ML },
        { id: 'ridge', label: 'Ridge Regression', desc: 'L2 Regularization', mode: StudioMode.DECISION_BOUNDARY },
        { id: 'lasso', label: 'Lasso Regression', desc: 'L1 Regularization', mode: StudioMode.DECISION_BOUNDARY },
        { id: 'elasticnet', label: 'ElasticNet', desc: 'L1 + L2 Regularization', mode: StudioMode.DECISION_BOUNDARY }
      ]
    },
    {
        id: 'ENSEMBLE',
        title: 'Ensemble Methods',
        icon: <Layers size={24} />,
        description: 'Boosting, Bagging, and Stacking.',
        models: [
            { id: 'random-forest', label: 'Random Forest', desc: 'Bagging of Trees', mode: StudioMode.GENERAL_ML },
            { id: 'xgboost', label: 'XGBoost', desc: 'Extreme Gradient Boosting', mode: StudioMode.ENSEMBLE_METHODS },
            { id: 'lightgbm', label: 'LightGBM', desc: 'Leaf-wise Tree Growth', mode: StudioMode.ENSEMBLE_METHODS },
            { id: 'gbm', label: 'GBM', desc: 'Gradient Boosting Machine', mode: StudioMode.ENSEMBLE_METHODS },
            { id: 'adaboost', label: 'AdaBoost', desc: 'Adaptive Boosting', mode: StudioMode.ENSEMBLE_METHODS },
            { id: 'catboost', label: 'CatBoost', desc: 'Categorical Boosting', mode: StudioMode.ENSEMBLE_METHODS }
        ]
    },
    {
      id: 'UNSUPERVISED',
      title: 'Unsupervised & Dim. Red.',
      icon: <Database size={24} />,
      description: 'Clustering, PCA, and Manifold Learning.',
      models: [
        { id: 'kmeans', label: 'K-Means', desc: 'Centroid Clustering', mode: StudioMode.CLUSTERING },
        { id: 'dbscan', label: 'DBSCAN', desc: 'Density Clustering', mode: StudioMode.CLUSTERING },
        { id: 'gmm', label: 'Gaussian Mixture', desc: 'Probabilistic Clustering', mode: StudioMode.CLUSTERING },
        { id: 'hierarchical', label: 'Hierarchical', desc: 'Agglomerative Clustering', mode: StudioMode.CLUSTERING },
        { id: 'pca', label: 'PCA', desc: 'Principal Component Analysis', mode: StudioMode.DIMENSIONALITY_REDUCTION },
        { id: 'lda', label: 'LDA', desc: 'Linear Discriminant Analysis', mode: StudioMode.DIMENSIONALITY_REDUCTION },
        { id: 'tsne', label: 't-SNE', desc: 't-Distributed Stochastic Neighbor', mode: StudioMode.DIMENSIONALITY_REDUCTION },
        { id: 'umap', label: 'UMAP', desc: 'Uniform Manifold Approximation', mode: StudioMode.DIMENSIONALITY_REDUCTION }
      ]
    },
    {
      id: 'AUDIO',
      title: 'Audio & Time Series',
      icon: <Mic size={24} />,
      description: 'Speech processing and temporal forecasting.',
      models: [
        { id: 'whisper', label: 'Whisper', desc: 'Speech Recognition', mode: StudioMode.AUDIO_PROCESSING },
        { id: 'wav2vec', label: 'Wav2Vec 2.0', desc: 'Self-supervised Audio', mode: StudioMode.AUDIO_PROCESSING },
        { id: 'sarima', label: 'SARIMA', desc: 'Seasonal ARIMA', mode: StudioMode.GENERAL_ML },
        { id: 'holt-winters', label: 'Holt-Winters', desc: 'Exponential Smoothing', mode: StudioMode.GENERAL_ML }
      ]
    },
    {
        id: 'DATA_ENG',
        title: 'Data Engineering',
        icon: <Fingerprint size={24} />,
        description: 'Preprocessing, Augmentation, and Features.',
        models: [
            { id: 'smote', label: 'SMOTE', desc: 'Synthetic Minority Oversampling', mode: StudioMode.DATA_PREPROCESSING },
            { id: 'augmentation', label: 'Data Augmentation', desc: 'Image/Text Transformations', mode: StudioMode.DATA_PREPROCESSING },
            { id: 'normalization', label: 'Scaling/Norm', desc: 'MinMax, StandardScaler', mode: StudioMode.DATA_PREPROCESSING },
            { id: 'feature-selection', label: 'Feature Selection', desc: 'RFE, Chi-Square', mode: StudioMode.FEATURE_ENGINEERING }
        ]
    },
    {
      id: 'RL_OPT',
      title: 'RL & Optimization',
      icon: <Activity size={24} />,
      description: 'Agents and Loss Landscapes.',
      models: [
        { id: 'q-learning', label: 'Q-Learning', desc: 'Value-based RL', mode: StudioMode.RL_ARENA },
        { id: 'loss-surface', label: 'Loss Landscape', desc: 'Gradient Descent Viz', mode: StudioMode.LOSS_LANDSCAPE },
        { id: 'sgd', label: 'SGD', desc: 'Stochastic Gradient Descent', mode: StudioMode.LOSS_LANDSCAPE },
        { id: 'adam', label: 'Adam Optimizer', desc: 'Adaptive Moment Estimation', mode: StudioMode.LOSS_LANDSCAPE }
      ]
    }
];

const DEFAULT_VISUALS: VisualSettings = {
    theme: 'cyber', showGrid: true, showStars: true, autoRotateSpeed: 0.3, glowIntensity: 1.5,
    particleDensity: 0.8, signalSpeed: 2.5, wireframe: false, connectionOpacity: 0.3,
    showClusterHulls: true, showCentroids: true, showSupportVectors: true,
    decisionSurfaceOpacity: 0.2, lossWaterLevel: -10, lossHeightScale: 1.5,
    showContourLines: true, neuronSize: 0.5
};

const DEFAULT_NEURAL: MLModelConfig = {
  mode: StudioMode.NEURAL_ARCH,
  layers: [
    { id: 'l1', type: LayerType.INPUT, neurons: 784, activationFunction: ActivationType.LINEAR, name: 'Input Buffer' },
    { id: 'l2', type: LayerType.CONV2D, neurons: 16, activationFunction: ActivationType.RELU, name: 'Conv Block A', featureMapSize: 4 },
    { id: 'l3', type: LayerType.POOLING, neurons: 16, activationFunction: ActivationType.LINEAR, name: 'Pool Layer' },
    { id: 'l4', type: LayerType.FLATTEN, neurons: 128, activationFunction: ActivationType.LINEAR, name: 'Flatten' },
    { id: 'l5', type: LayerType.DENSE, neurons: 64, activationFunction: ActivationType.RELU, name: 'Dense Core' },
    { id: 'l6', type: LayerType.OUTPUT, neurons: 10, activationFunction: ActivationType.SOFTMAX, name: 'Output Class' },
  ],
  dropoutRate: 0.2, optimizer: OptimizerType.ADAM, learningRate: 0.001, batchSize: 64, epochs: 20, noise: 1.0, connectionSparsity: 1.0
};

// Helper to get quick configs without API calls for standard models
const getPresetConfig = (modelId: string, mode: StudioMode): MLModelConfig => {
    const base: MLModelConfig = { ...DEFAULT_NEURAL, mode };
    
    switch(modelId) {
        // --- GEN AI ---
        case 'stable-diffusion':
        case 'diffusion':
            return {
                ...base,
                mode: StudioMode.GENERATIVE_AI,
                layers: [
                    { id: 'lat', type: LayerType.INPUT, neurons: 64, activationFunction: ActivationType.LINEAR, name: 'Latent Space' },
                    { id: 'unet_down', type: LayerType.CONV2D, neurons: 128, activationFunction: ActivationType.RELU, name: 'U-Net Down' },
                    { id: 'attn', type: LayerType.DENSE, neurons: 256, activationFunction: ActivationType.GELU, name: 'Cross-Attention' },
                    { id: 'unet_up', type: LayerType.CONV2D, neurons: 128, activationFunction: ActivationType.RELU, name: 'U-Net Up' },
                    { id: 'out', type: LayerType.OUTPUT, neurons: 64, activationFunction: ActivationType.LINEAR, name: 'Denoised' },
                ]
            };
        case 'gan':
             return {
                 ...base,
                 mode: StudioMode.GENERATIVE_AI,
                 layers: [
                     { id: 'z', type: LayerType.INPUT, neurons: 100, activationFunction: ActivationType.LINEAR, name: 'Latent Z' },
                     { id: 'g_hidden', type: LayerType.DENSE, neurons: 256, activationFunction: ActivationType.LEAKY_RELU, name: 'Generator' },
                     { id: 'fake', type: LayerType.OUTPUT, neurons: 784, activationFunction: ActivationType.TANH, name: 'Fake Image' },
                     { id: 'd_hidden', type: LayerType.DENSE, neurons: 256, activationFunction: ActivationType.LEAKY_RELU, name: 'Discriminator' },
                     { id: 'validity', type: LayerType.OUTPUT, neurons: 1, activationFunction: ActivationType.SIGMOID, name: 'Real/Fake' },
                 ]
             };
        case 'gpt':
        case 'llama':
        case 'bert':
        case 't5':
            return {
                ...base,
                mode: StudioMode.NLP,
                layers: [
                    { id: 'emb', type: LayerType.INPUT, neurons: 768, activationFunction: ActivationType.LINEAR, name: 'Embeddings' },
                    { id: 'mha', type: LayerType.DENSE, neurons: 768, activationFunction: ActivationType.LINEAR, name: 'Multi-Head Attn' },
                    { id: 'add_norm', type: LayerType.BATCH_NORM, neurons: 768, activationFunction: ActivationType.LINEAR, name: 'Add & Norm' },
                    { id: 'ffn', type: LayerType.DENSE, neurons: 3072, activationFunction: ActivationType.GELU, name: 'Feed Forward' },
                    { id: 'out', type: LayerType.OUTPUT, neurons: 50000, activationFunction: ActivationType.SOFTMAX, name: 'Logits' },
                ]
            };
            
        // --- COMPUTER VISION ---
        case 'yolo':
        case 'cnn-standard':
        case 'detr':
            return {
                ...base,
                mode: StudioMode.COMPUTER_VISION,
                layers: [
                     { id: 'in', type: LayerType.INPUT, neurons: 1024, activationFunction: ActivationType.LINEAR, name: 'Input RGB' },
                     { id: 'backbone', type: LayerType.CONV2D, neurons: 64, activationFunction: ActivationType.RELU, name: 'Backbone' },
                     { id: 'neck', type: LayerType.CONV2D, neurons: 128, activationFunction: ActivationType.RELU, name: 'FPN Neck' },
                     { id: 'head', type: LayerType.DENSE, neurons: 80, activationFunction: ActivationType.SIGMOID, name: 'Detection Head' },
                ]
            };
        case 'vit':
        case 'clip':
             return {
                 ...base,
                 mode: StudioMode.COMPUTER_VISION,
                 layers: [
                     { id: 'patch', type: LayerType.INPUT, neurons: 196, activationFunction: ActivationType.LINEAR, name: 'Patch Embed' },
                     { id: 'enc', type: LayerType.DENSE, neurons: 768, activationFunction: ActivationType.GELU, name: 'Transformer Enc' },
                     { id: 'cls', type: LayerType.OUTPUT, neurons: 1000, activationFunction: ActivationType.SOFTMAX, name: 'Class Token' },
                 ]
             };

        // --- CLASSICAL / ENSEMBLE ---
        case 'svm':
            return { mode: StudioMode.DECISION_BOUNDARY, kernel: 'rbf', cParameter: 1.0 };
        case 'linear-reg':
        case 'ridge':
        case 'lasso':
        case 'elasticnet':
            return { mode: StudioMode.DECISION_BOUNDARY, kernel: 'linear' };
        case 'random-forest':
        case 'xgboost':
        case 'lightgbm':
        case 'catboost':
        case 'adaboost':
            return { mode: StudioMode.GENERAL_ML, algorithm: 'RandomForest', nEstimators: 20, treeDepth: 6 };
        case 'naive-bayes':
             return { mode: StudioMode.GENERAL_ML, algorithm: 'DecisionTree', treeDepth: 1 }; // Viz approximation
        case 'knn':
             return { mode: StudioMode.GENERAL_ML, algorithm: 'KNN', nNeighbors: 5 };
        case 'decision-tree':
             return { mode: StudioMode.GENERAL_ML, algorithm: 'DecisionTree', treeDepth: 4 };

        // --- UNSUPERVISED ---
        case 'kmeans':
            return { mode: StudioMode.CLUSTERING, k: 5, pointsCount: 500, clusteringAlgorithm: 'KMeans', distanceMetric: 'Euclidean' };
        case 'dbscan':
            return { mode: StudioMode.CLUSTERING, k: 0, pointsCount: 600, clusteringAlgorithm: 'DBSCAN', distanceMetric: 'Euclidean', noise: 0.5 };
        case 'gmm':
             return { mode: StudioMode.CLUSTERING, k: 3, pointsCount: 400, clusteringAlgorithm: 'KMeans' }; // Approximation
        case 'pca':
        case 'lda':
        case 'tsne':
        case 'umap':
            return { mode: StudioMode.DIMENSIONALITY_REDUCTION };

        // --- RL & OPT ---
        case 'loss-surface':
        case 'sgd':
        case 'adam':
             return { mode: StudioMode.LOSS_LANDSCAPE, landscapeType: 'non-convex', resolution: 64 };
             
        default:
            return base;
    }
}

// --- SUB-COMPONENTS ---

const TermsModal = ({ onClose }: { onClose: () => void }) => (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-fade-in">
        <div className="w-full max-w-2xl glass-panel-pro rounded-2xl flex flex-col max-h-[85vh] shadow-[0_0_50px_rgba(0,0,0,0.7)] border-t border-white/10">
            <div className="p-6 border-b border-white/5 flex justify-between items-center bg-white/5 rounded-t-2xl">
                <div className="flex items-center gap-3">
                    <Shield className="text-cyber-accent" size={20} />
                    <h2 className="text-sm font-display font-bold text-white tracking-widest uppercase">TERMS_OF_SERVICE</h2>
                </div>
                <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors"><X size={20}/></button>
            </div>
            
            <div className="p-8 overflow-y-auto custom-scrollbar space-y-6">
                <div className="space-y-2">
                    <h3 className="text-xs font-bold text-cyber-accent uppercase tracking-wider">1. Access & Usage Protocol</h3>
                    <p className="text-xs text-gray-400 font-mono leading-relaxed">
                        By initializing the Dracarys AI interface ("The Platform"), you agree to utilize the neural visualization engines for educational, analytical, and research purposes only. Reverse engineering of the WebGL rendering core or unauthorized API injection is strictly prohibited under cyber-security protocols.
                    </p>
                </div>

                <div className="space-y-2">
                    <h3 className="text-xs font-bold text-cyber-accent uppercase tracking-wider">2. Generative AI Disclaimer</h3>
                    <p className="text-xs text-gray-400 font-mono leading-relaxed">
                        This platform integrates Google Gemini APIs for architectural synthesis and code generation. Outputs are probabilistic in nature. Dracarys Systems does not guarantee the convergence or production-readiness of generated models in real-world training scenarios. Always verify generated PyTorch/TensorFlow code before deployment.
                    </p>
                </div>

                <div className="space-y-2">
                    <h3 className="text-xs font-bold text-cyber-accent uppercase tracking-wider">3. Data Privacy & Compute</h3>
                    <p className="text-xs text-gray-400 font-mono leading-relaxed">
                        Input data for clustering, regression, and manifold visualization is processed client-side via WebGL shaders. No raw datasets are transmitted to external servers. Architectural queries sent to the LLM are subject to Google's API data usage policies.
                    </p>
                </div>

                <div className="space-y-2">
                    <h3 className="text-xs font-bold text-cyber-accent uppercase tracking-wider">4. Limitation of Liability</h3>
                    <p className="text-xs text-gray-400 font-mono leading-relaxed">
                        The Platform is provided "AS IS", without warranty of any kind. Dracarys AI is not liable for GPU thermal throttling, browser crashes due to excessive particle density settings, or loss of unsaved architectural blueprints.
                    </p>
                </div>
            </div>

            <div className="p-6 border-t border-white/5 flex justify-end bg-black/20 rounded-b-2xl">
                <Button onClick={onClose} variant="primary" size="sm" icon={<Check size={14}/>}>ACKNOWLEDGE PROTOCOL</Button>
            </div>
        </div>
    </div>
);

const DracarysLogo = ({ className = "w-40 h-40" }: { className?: string }) => (
    <div className={`relative group ${className} flex items-center justify-center animate-float`}>
        <div className="absolute inset-0 bg-cyber-accent/10 blur-3xl rounded-full opacity-0 group-hover:opacity-40 transition-opacity duration-1000"></div>
        <img 
            src="./logo.png" 
            alt="Dracarys AI Logo" 
            className="relative z-10 w-full h-full object-contain filter drop-shadow-[0_0_15px_rgba(0,240,255,0.2)] transition-transform duration-500 group-hover:scale-105"
            onError={(e) => {
                e.currentTarget.style.display = 'none';
                e.currentTarget.parentElement!.innerHTML += `<div class="text-[10px] text-cyber-accent border border-cyber-accent px-2 py-1 rounded">LOGO</div>`;
            }}
        />
    </div>
);

const CategoryCard: React.FC<{ item: ModelCategory, onClick: () => void }> = ({ item, onClick }) => (
    <div onClick={onClick} className="hover-card-trigger relative cursor-pointer group perspective-1000">
        <div className="card-content h-48 bg-cyber-900/40 backdrop-blur-md border border-white/5 rounded-2xl p-6 flex flex-col justify-between transition-all duration-300 ease-out relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-cyber-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            
            <div className="flex justify-between items-start z-10">
                <div className="p-3 bg-white/5 rounded-xl text-cyber-accent border border-white/5 group-hover:bg-cyber-accent group-hover:text-black transition-colors duration-300 shadow-lg">
                    {item.icon}
                </div>
                <div className="w-8 h-8 rounded-full border border-white/10 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all -translate-x-2 group-hover:translate-x-0">
                    <ChevronRight size={14} />
                </div>
            </div>
            
            <div className="z-10 space-y-2">
                <h4 className="font-display font-bold text-lg text-white group-hover:text-cyber-accent transition-colors">{item.title}</h4>
                <p className="text-xs text-gray-400 font-sans leading-relaxed group-hover:text-gray-300">{item.description}</p>
            </div>
        </div>
    </div>
);

const ModelCard: React.FC<{ item: ModelDefinition, onClick: () => void }> = ({ item, onClick }) => (
    <button onClick={onClick} className="group relative text-left w-full h-full">
         <div className="absolute inset-0 bg-cyber-accent/5 rounded-xl blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
         <div className="relative h-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyber-accent/50 rounded-xl p-5 transition-all duration-300 flex flex-col gap-3">
             <div className="flex items-center justify-between">
                 <span className="text-xs font-mono text-cyber-accent opacity-70 group-hover:opacity-100">{item.id.toUpperCase()}</span>
                 <ArrowLeft className="rotate-180 text-gray-500 group-hover:text-cyber-accent transition-colors" size={14} />
             </div>
             <div>
                 <h4 className="font-bold text-white text-sm group-hover:text-cyber-accent transition-colors">{item.label}</h4>
                 <p className="text-[10px] text-gray-400 mt-1 leading-relaxed">{item.desc}</p>
             </div>
         </div>
    </button>
);

const OscilloscopeGraph = ({ data, color, label }: { data: number[], color: string, label: string }) => {
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = max - min || 1;
    return (
        <div className="relative h-24 bg-black/20 rounded-lg overflow-hidden flex flex-col justify-end group border border-white/5">
             <div className="absolute top-2 left-3 text-[10px] font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                 <div className="w-1.5 h-1.5 rounded-full" style={{backgroundColor: color, boxShadow: `0 0 8px ${color}`}}></div>
                 {label}
             </div>
             <svg className="absolute inset-0 w-full h-full p-2 pt-6" preserveAspectRatio="none">
                 <path d={`M 0 100 L ${data.map((v, i) => `${(i / (data.length - 1)) * 100} ${100 - ((v - min) / range) * 80}`).join(' L ')} L 100 100 Z`} fill={color} fillOpacity="0.1" />
                 <path d={`M ${data.map((v, i) => `${(i / (data.length - 1)) * 100} ${100 - ((v - min) / range) * 80}`).join(' L ')}`} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
             </svg>
             <div className="absolute bottom-2 right-3 text-xs font-mono font-bold text-white">{data[data.length-1]?.toFixed(3)}</div>
        </div>
    )
}

const RangeControl = ({ label, value, min, max, step, onChange }: { label: string, value: number, min: number, max: number, step: number, onChange: (v: number) => void }) => (
    <div className="group space-y-2">
        <div className="flex justify-between text-xs font-medium text-gray-400 group-hover:text-white transition-colors">
            <span>{label}</span>
            <span className="font-mono text-cyber-accent">{value.toFixed(step < 0.1 ? 3 : 1)}</span>
        </div>
        <div className="relative h-1.5 bg-white/10 rounded-full w-full overflow-hidden">
            <div className="absolute h-full bg-cyber-accent rounded-full shadow-[0_0_10px_#00f0ff]" style={{width: `${((value-min)/(max-min))*100}%`}}></div>
            <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
        </div>
    </div>
);

// --- MAIN APP ---

const App: React.FC = () => {
  const [booted, setBooted] = useState(false);
  const [view, setView] = useState<'HOME' | 'STUDIO'>('HOME');
  const [activeMode, setActiveMode] = useState<StudioMode>(StudioMode.NEURAL_ARCH);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  const [modelConfig, setModelConfig] = useState<MLModelConfig>(DEFAULT_NEURAL);
  const [visuals, setVisuals] = useState<VisualSettings>(DEFAULT_VISUALS);
  
  const [activePanel, setActivePanel] = useState<'config' | 'visuals' | 'code' | 'export' | null>(null);
  const [showTerms, setShowTerms] = useState(false);
  
  // Floating Window States
  const [panels, setPanels] = useState({ telemetry: true, trace: true });
  
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [vizState, setVizState] = useState<VizState>({ isPlaying: true, epoch: 0, metric1: 0.5, metric2: 0.5, metric1Name: 'ERR', metric2Name: 'ACC', tp: 50, tn: 40, fp: 5, fn: 5, history1: Array(30).fill(0.5), history2: Array(30).fill(0.5) });
  
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [algoSteps, setAlgoSteps] = useState<AlgorithmStep[]>([]);
  const [stepsLoading, setStepsLoading] = useState(false);
  const [codeLang, setCodeLang] = useState<'pytorch' | 'tensorflow'>('pytorch');
  const threeSceneRef = useRef<ThreeSceneHandle>(null);

  const addToast = (type: Toast['type'], message: string) => {
    const id = Date.now().toString(); setToasts(prev => [...prev, { id, type, message }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  };

  const handleSpecificModelSelect = (modelDef: ModelDefinition) => {
      setActiveMode(modelDef.mode);
      setVizState(s => ({ ...s, epoch: 0, history1: [], history2: [] }));
      
      const config = getPresetConfig(modelDef.id, modelDef.mode);
      
      // Update Metrics Naming based on mode
      if (modelDef.mode === StudioMode.CLUSTERING) {
          setVizState(prev => ({...prev, metric1Name: 'INERTIA', metric2Name: 'SILHOUETTE'}));
      } else if (modelDef.mode === StudioMode.RL_ARENA) {
          setVizState(prev => ({...prev, metric1Name: 'REWARD', metric2Name: 'LOSS'}));
      } else {
          setVizState(prev => ({...prev, metric1Name: 'LOSS', metric2Name: 'ACCURACY'}));
      }
      
      setModelConfig(config);
      setView('STUDIO');
      addToast('success', `LOADED: ${modelDef.label.toUpperCase()}`);
  };

  const handleModelSearch = async (modelName: string) => {
    if (!modelName.trim()) return;
    setIsGenerating(true);
    addToast('info', `ARCHITECTING "${modelName.toUpperCase()}"...`);
    const newConfig = await generateConfigFromModelName(modelName);
    if (newConfig) {
        if (newConfig.layers) newConfig.layers = newConfig.layers.map((l, i) => ({...l, id: l.id || `gen_l${i}`}));
        setModelConfig(newConfig); setActiveMode(newConfig.mode);
        setView('STUDIO');
        addToast('success', `BLUEPRINT GENERATED`);
    } else addToast('error', `GENERATION FAILED`);
    setIsGenerating(false);
  };

  useEffect(() => { const load = async () => { setStepsLoading(true); setAlgoSteps(await generateSteps(modelConfig)); setStepsLoading(false); }; load(); }, [modelConfig]);

  useEffect(() => {
    let interval: any;
    if (vizState.isPlaying && view === 'STUDIO') {
      interval = setInterval(() => {
        setVizState(prev => {
          const newM1 = Math.max(0.01, prev.metric1 * 0.99 - Math.random() * 0.01);
          const newM2 = Math.min(0.99, prev.metric2 + Math.random() * 0.005);
          return {
            ...prev, epoch: prev.epoch + 1, metric1: newM1, metric2: newM2,
            history1: [...prev.history1, newM1].slice(-30), history2: [...prev.history2, newM2].slice(-30),
            tp: Math.min(100, prev.tp + (Math.random() > 0.5 ? 1 : 0)),
          };
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [vizState.isPlaying, view]);

  // Layer Management Logic
  const updateLayer = (id: string, updates: any) => modelConfig.layers && setModelConfig({...modelConfig, layers: modelConfig.layers.map(l => l.id === id ? {...l, ...updates} : l)});
  const toggleLayer = (id: string) => modelConfig.layers && setModelConfig({...modelConfig, layers: modelConfig.layers.filter(l => l.id !== id)});
  const addLayer = () => { if(!modelConfig.layers) return; const id = `l_${Date.now()}`; setModelConfig({...modelConfig, layers: [...modelConfig.layers.slice(0, -1), { id, type: LayerType.DENSE, neurons: 16, activationFunction: ActivationType.RELU, name: 'Dense' }, modelConfig.layers[modelConfig.layers.length-1]]}); };

  const getGeneratedCode = () => {
    if (codeLang === 'pytorch') {
      if (modelConfig.mode === StudioMode.NEURAL_ARCH && modelConfig.layers) {
        let code = `import torch\nimport torch.nn as nn\n\nclass CustomModel(nn.Module):\n    def __init__(self):\n        super(CustomModel, self).__init__()\n        self.layers = nn.Sequential(\n`;
        modelConfig.layers.forEach(layer => {
             if (layer.type === LayerType.CONV2D) code += `            nn.Conv2d(in_channels=?, out_channels=${layer.neurons}, kernel_size=3),\n`;
             if (layer.type === LayerType.DENSE) code += `            nn.Linear(in_features=?, out_features=${layer.neurons}),\n`;
             if (layer.type === LayerType.POOLING) code += `            nn.MaxPool2d(2),\n`;
             if (layer.type === LayerType.FLATTEN) code += `            nn.Flatten(),\n`;
             if (layer.type === LayerType.DROPOUT) code += `            nn.Dropout(p=${modelConfig.dropoutRate || 0.5}),\n`;
             if (layer.activationFunction === ActivationType.RELU) code += `            nn.ReLU(),\n`;
             if (layer.activationFunction === ActivationType.SOFTMAX) code += `            nn.Softmax(dim=1),\n`;
        });
        code += `        )\n\n    def forward(self, x):\n        return self.layers(x)`;
        return code;
      }
      return `# PyTorch code generation for ${modelConfig.mode} not yet implemented.`;
    } else {
        // Tensorflow/Keras
      if (modelConfig.mode === StudioMode.NEURAL_ARCH && modelConfig.layers) {
          let code = `import tensorflow as tf\nfrom tensorflow.keras import layers, models\n\nmodel = models.Sequential([\n`;
           modelConfig.layers.forEach(layer => {
             if (layer.type === LayerType.CONV2D) code += `    layers.Conv2D(${layer.neurons}, (3, 3), activation='${layer.activationFunction.toLowerCase()}'),\n`;
             else if (layer.type === LayerType.DENSE) code += `    layers.Dense(${layer.neurons}, activation='${layer.activationFunction.toLowerCase()}'),\n`;
             else if (layer.type === LayerType.POOLING) code += `    layers.MaxPooling2D((2, 2)),\n`;
             else if (layer.type === LayerType.FLATTEN) code += `    layers.Flatten(),\n`;
             else if (layer.type === LayerType.DROPOUT) code += `    layers.Dropout(${modelConfig.dropoutRate || 0.5}),\n`;
             else if (layer.type === LayerType.OUTPUT) code += `    layers.Dense(${layer.neurons}, activation='softmax'),\n`;
           });
          code += `])\n\nmodel.compile(optimizer='${modelConfig.optimizer || 'adam'}',\n              loss='sparse_categorical_crossentropy',\n              metrics=['accuracy'])`;
          return code;
      }
      return `# TensorFlow code generation for ${modelConfig.mode} not yet implemented.`;
    }
  };

  if (!booted) {
      return (
        <div className="fixed inset-0 bg-black z-[100] flex flex-col items-center justify-center font-mono">
            <div className="animate-pulse mb-8">
                 <DracarysLogo className="w-32 h-32" />
            </div>
            <div className="w-64 h-1 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-cyber-accent animate-[slide-in-right_2s_ease-out_forwards]"></div>
            </div>
            <div className="mt-4 text-xs text-cyber-accent tracking-[0.3em] opacity-80">INITIALIZING CORE</div>
            {/* Self-boot for demo */}
            {setTimeout(() => setBooted(true), 2500) && null} 
        </div>
      )
  }

  const selectedCategoryData = MODEL_CATEGORIES.find(c => c.id === selectedCategory);

  return (
    <div className="flex flex-col h-screen w-full text-gray-200 font-sans overflow-hidden relative selection:bg-cyber-accent selection:text-cyber-950">
      
      {/* --- GLOBAL TOASTS --- */}
      <div className="absolute top-24 right-8 flex flex-col gap-3 z-[70] pointer-events-none">
        {toasts.map(t => (
          <div key={t.id} className={`pointer-events-auto w-80 p-4 glass-panel-pro rounded-xl animate-slide-in-right flex gap-4 items-center border-l-4 ${t.type === 'success' ? 'border-green-500' : t.type === 'error' ? 'border-red-500' : 'border-cyber-accent'}`}>
            <div className={`${t.type==='success'?'text-green-400':t.type==='error'?'text-red-400':'text-cyber-accent'}`}>
                 {t.type === 'success' ? <Check size={20} /> : t.type === 'error' ? <AlertCircle size={20} /> : <Info size={20} />}
            </div>
            <div>
                <h4 className="text-xs font-bold uppercase tracking-wider opacity-60">{t.type}</h4>
                <p className="text-sm font-medium text-white">{t.message}</p>
            </div>
          </div>
        ))}
      </div>

      {/* --- TERMS MODAL --- */}
      {showTerms && <TermsModal onClose={() => setShowTerms(false)} />}

      {/* --- HOME DASHBOARD --- */}
      {view === 'HOME' && (
          <main className="flex-1 relative z-10 overflow-y-auto custom-scrollbar">
              <div className="max-w-7xl mx-auto px-6 py-12 md:py-20 flex flex-col gap-16">
                  
                  {/* HERO SECTION */}
                  <div className="flex flex-col md:flex-row items-center justify-between gap-12 animate-slide-up">
                      <div className="flex-1 space-y-8">
                          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyber-accent/10 border border-cyber-accent/20 text-cyber-accent text-xs font-mono tracking-wider">
                              <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyber-accent opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyber-accent"></span>
                              </span>
                              SYSTEM ONLINE v2.4.0
                          </div>
                          
                          <div className="space-y-4">
                              <h1 className="text-6xl md:text-8xl font-display font-bold text-white tracking-tight leading-none">
                                  PROJECT <br/> <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyber-accent to-cyber-secondary text-glow">DRACARYS</span>
                              </h1>
                              <p className="text-lg text-gray-400 max-w-xl leading-relaxed">
                                  The enterprise standard for 3D Machine Learning visualization. Architect neural topologies, explore high-dimensional manifolds, and optimize decision boundaries in real-time.
                              </p>
                          </div>

                          <div className="relative max-w-lg">
                              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                  <Search className="text-gray-500" size={20}/>
                              </div>
                              <input 
                                  type="text" 
                                  placeholder="Describe a model (e.g. 'ResNet-50', 'Transformer', 'PCA')..." 
                                  className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-12 pr-12 text-white placeholder-gray-500 focus:outline-none focus:border-cyber-accent/50 focus:bg-white/10 transition-all font-mono text-sm shadow-xl"
                                  onKeyDown={(e) => { if (e.key === 'Enter' && !isGenerating) { handleModelSearch(e.currentTarget.value); e.currentTarget.value = ''; } }} 
                              />
                              {isGenerating && <div className="absolute inset-y-0 right-4 flex items-center"><Loader2 className="animate-spin text-cyber-accent" size={20}/></div>}
                          </div>
                      </div>
                      
                      <div className="flex-1 flex justify-center relative">
                           {/* Decorative rings */}
                           <div className="absolute w-[500px] h-[500px] border border-white/5 rounded-full animate-spin-slow"></div>
                           <div className="absolute w-[350px] h-[350px] border border-white/5 rounded-full animate-spin-slow" style={{animationDirection: 'reverse'}}></div>
                           <DracarysLogo className="w-64 h-64 md:w-96 md:h-96" />
                      </div>
                  </div>

                  {/* MODULE GRID - LEVEL 1: CATEGORIES */}
                  {!selectedCategory && (
                      <div className="space-y-8 animate-slide-up" style={{animationDelay: '100ms'}}>
                          <div className="flex items-end justify-between border-b border-white/10 pb-4">
                              <h2 className="text-2xl font-display font-bold text-white flex items-center gap-3">
                                  <Layers className="text-cyber-accent" /> Intelligence Domains
                              </h2>
                          </div>
                          
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-20">
                              {MODEL_CATEGORIES.map(category => (
                                  <CategoryCard key={category.id} item={category} onClick={() => setSelectedCategory(category.id)} />
                              ))}
                          </div>
                      </div>
                  )}

                  {/* MODULE GRID - LEVEL 2: SPECIFIC MODELS */}
                  {selectedCategory && selectedCategoryData && (
                      <div className="space-y-8 animate-slide-up bg-black/40 -mx-6 px-6 py-8 rounded-2xl border-y border-white/5">
                          <div className="flex items-center gap-4 border-b border-white/10 pb-4 mb-8">
                              <button onClick={() => setSelectedCategory(null)} className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-white transition-colors">
                                  <ChevronLeft size={20} />
                              </button>
                              <div className="flex flex-col">
                                  <h2 className="text-2xl font-display font-bold text-white flex items-center gap-3">
                                      {selectedCategoryData.icon} {selectedCategoryData.title}
                                  </h2>
                                  <span className="text-xs text-cyber-accent uppercase tracking-widest">Select Architecture Protocol</span>
                              </div>
                          </div>
                          
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pb-4">
                              {selectedCategoryData.models.map(model => (
                                  <div key={model.id} className="h-40">
                                      <ModelCard item={model} onClick={() => handleSpecificModelSelect(model)} />
                                  </div>
                              ))}
                          </div>
                      </div>
                  )}

                  {/* FOOTER */}
                  <div className="mt-8 pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4 text-[10px] text-gray-500 font-mono pb-8">
                       <span className="opacity-50">© 2025 DRACARYS SYSTEMS. ALL RIGHTS RESERVED.</span>
                       <div className="flex gap-6">
                           <button onClick={() => setShowTerms(true)} className="hover:text-cyber-accent transition-colors">TERMS & CONDITIONS</button>
                           <button onClick={() => addToast('info', 'PRIVACY_PROTOCOL_ACTIVE')} className="hover:text-cyber-accent transition-colors">PRIVACY PROTOCOL</button>
                       </div>
                  </div>
              </div>
          </main>
      )}

      {/* --- STUDIO VIEW --- */}
      {view === 'STUDIO' && (
          <div className="flex-1 flex relative overflow-hidden bg-black/80">
             
             {/* 3D Environment */}
             <div className="absolute inset-0 z-0">
                 <Suspense fallback={<div className="w-full h-full flex items-center justify-center"><Loader2 className="w-12 h-12 animate-spin text-cyber-accent"/></div>}>
                    <ThreeScene ref={threeSceneRef} config={modelConfig} visuals={visuals} isPlaying={vizState.isPlaying} />
                 </Suspense>
             </div>

             {/* Top Status Bar (Minimal) */}
             <header className="absolute top-6 left-1/2 -translate-x-1/2 z-30 glass-panel-pro px-6 py-2 rounded-full flex items-center gap-8 shadow-2xl">
                 <div className="flex items-center gap-3">
                    <button onClick={() => setView('HOME')} className="text-gray-400 hover:text-white transition-colors">
                         <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
                            <ArrowLeft size={14} /> 
                         </div>
                    </button>
                    <div className="h-4 w-px bg-white/10"></div>
                    <span className="font-display font-bold text-white tracking-widest text-sm">{activeMode.replace('_', ' ')}</span>
                 </div>
                 
                 <div className="hidden md:flex items-center gap-4 text-xs font-mono text-gray-400">
                     <div className="flex items-center gap-2"><Cpu size={12} className="text-cyber-accent"/> GPU: OPTIMAL</div>
                     <div className="flex items-center gap-2"><Activity size={12} className="text-cyber-secondary"/> EPOCH: {vizState.epoch}</div>
                 </div>
             </header>

             {/* FLOATING WINDOW: Telemetry */}
             {panels.telemetry && (
                 <div className="absolute left-6 top-24 w-80 z-20 glass-panel-pro rounded-xl flex flex-col shadow-2xl animate-fade-in border-l-2 border-l-cyber-accent">
                      <div className="h-10 flex items-center justify-between px-4 border-b border-white/5 bg-white/5">
                          <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2"><Activity size={14} className="text-cyber-accent"/> Telemetry</span>
                          <button onClick={() => setPanels({...panels, telemetry: false})}><X size={14} className="text-gray-500 hover:text-white"/></button>
                      </div>
                      <div className="p-4 space-y-4">
                           <OscilloscopeGraph data={vizState.history1} color="#ef4444" label={vizState.metric1Name} />
                           <OscilloscopeGraph data={vizState.history2} color="#00f0ff" label={vizState.metric2Name} />
                           <div className="grid grid-cols-2 gap-3 pt-2">
                               <div className="bg-white/5 rounded p-2 text-center border border-white/5">
                                   <div className="text-[10px] text-gray-400">PRECISION</div>
                                   <div className="text-lg font-mono font-bold text-green-400">98.2%</div>
                               </div>
                               <div className="bg-white/5 rounded p-2 text-center border border-white/5">
                                   <div className="text-[10px] text-gray-400">RECALL</div>
                                   <div className="text-lg font-mono font-bold text-cyber-accent">94.8%</div>
                               </div>
                           </div>
                      </div>
                 </div>
             )}

             {/* FLOATING WINDOW: Trace Logic */}
             {panels.trace && (
                 <div className="absolute right-6 top-24 w-80 z-20 glass-panel-pro rounded-xl flex flex-col shadow-2xl animate-fade-in border-r-2 border-r-cyber-secondary">
                      <div className="h-10 flex items-center justify-between px-4 border-b border-white/5 bg-white/5">
                          <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2"><Terminal size={14} className="text-cyber-secondary"/> Trace</span>
                          <button onClick={() => setPanels({...panels, trace: false})}><X size={14} className="text-gray-500 hover:text-white"/></button>
                      </div>
                      <div className="p-4 max-h-[400px] overflow-y-auto custom-scrollbar space-y-4">
                          {stepsLoading ? <div className="text-center py-4"><Loader2 className="animate-spin mx-auto text-gray-500"/></div> : algoSteps.map((step, i) => (
                              <div key={i} className="relative pl-4 border-l border-white/10 hover:border-cyber-accent transition-colors pb-1">
                                  <div className="absolute -left-[5px] top-0 w-2.5 h-2.5 rounded-full bg-black border-2 border-gray-600"></div>
                                  <h5 className="text-xs font-bold text-white mb-1">{step.title}</h5>
                                  <p className="text-[10px] text-gray-400 leading-relaxed">{step.description}</p>
                              </div>
                          ))}
                      </div>
                 </div>
             )}

             {/* THE DOCK (Bottom Controls) */}
             <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-40">
                 <div className="glass-dock px-6 py-3 rounded-2xl flex items-center gap-6">
                     
                     {/* Playback Controls */}
                     <div className="flex items-center gap-2 pr-6 border-r border-white/10">
                         <button onClick={() => setVizState(s => ({...s, isPlaying: !s.isPlaying}))} className="w-12 h-12 rounded-xl bg-cyber-accent text-black flex items-center justify-center hover:scale-105 transition-transform shadow-[0_0_20px_rgba(0,240,255,0.4)]">
                             {vizState.isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-1"/>}
                         </button>
                         <button onClick={() => setVizState(s => ({...s, isPlaying: false, epoch: 0, history1: [], history2: []}))} className="w-10 h-10 rounded-xl bg-white/5 hover:bg-white/10 text-white flex items-center justify-center transition-colors">
                             <RefreshCw size={18} />
                         </button>
                     </div>

                     {/* Tools */}
                     <div className="flex items-center gap-4">
                         {[
                             { id: 'config', icon: <Sliders size={20}/>, label: 'Config' },
                             { id: 'visuals', icon: <Eye size={20}/>, label: 'View' },
                             { id: 'code', icon: <Code size={20}/>, label: 'Code' },
                             { id: 'export', icon: <Download size={20}/>, label: 'Export' },
                         ].map(tool => (
                             <button 
                                key={tool.id}
                                onClick={() => setActivePanel(activePanel === tool.id ? null : tool.id as any)}
                                className={`relative group p-3 rounded-xl transition-all ${activePanel === tool.id ? 'bg-white/10 text-cyber-accent shadow-[0_0_15px_rgba(255,255,255,0.1)]' : 'text-gray-400 hover:text-white hover:bg-white/5 hover:-translate-y-1'}`}
                             >
                                 {tool.icon}
                                 <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-2 py-1 bg-black/80 rounded text-[10px] opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap border border-white/10">{tool.label}</div>
                             </button>
                         ))}
                     </div>

                     <div className="h-8 w-px bg-white/10 mx-2"></div>

                     <button onClick={() => setChatOpen(!chatOpen)} className={`relative p-3 rounded-xl transition-all ${chatOpen ? 'bg-cyber-secondary text-white shadow-[0_0_20px_rgba(112,0,255,0.4)]' : 'text-gray-400 hover:text-cyber-secondary hover:bg-white/5'}`}>
                         <MessageSquare size={20} />
                         <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                     </button>
                 </div>
             </div>

             {/* POPUP PANEL: Configuration / Visuals / Code / Export */}
             {activePanel && (
                 <div className="absolute bottom-28 left-1/2 -translate-x-1/2 w-[500px] z-30 glass-panel-pro rounded-2xl p-6 animate-slide-up shadow-2xl overflow-hidden">
                     <div className="flex justify-between items-center mb-6">
                         <h3 className="text-sm font-bold font-display uppercase tracking-widest text-white">
                             {activePanel === 'config' ? 'Model Configuration' : activePanel === 'visuals' ? 'Viewport Settings' : activePanel === 'export' ? 'Artifact Export' : 'Source Generation'}
                         </h3>
                         <button onClick={() => setActivePanel(null)}><X size={16} className="text-gray-500 hover:text-white"/></button>
                     </div>

                     <div className="max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
                         {activePanel === 'config' && (
                             <div className="space-y-6">
                                 {modelConfig.layers && (
                                     <div className="space-y-3">
                                         {modelConfig.layers.map((l, i) => (
                                             <div key={l.id} className="bg-white/5 p-3 rounded-lg flex items-center gap-3 border border-white/5 group hover:border-cyber-accent/30 transition-colors">
                                                 <span className="text-xs font-mono text-gray-500 w-6">{i}</span>
                                                 <select value={l.type} onChange={e => updateLayer(l.id, {type: e.target.value})} className="bg-transparent text-xs font-bold text-white focus:outline-none uppercase w-24"><option value="DENSE">DENSE</option><option value="CONV2D">CONV2D</option><option value="POOLING">POOL</option><option value="DROPOUT">DROP</option><option value="FLATTEN">FLAT</option></select>
                                                 {l.type !== 'FLATTEN' && l.type !== 'POOLING' && (
                                                    <input type="number" value={l.neurons} onChange={e => updateLayer(l.id, {neurons: parseInt(e.target.value)})} className="bg-transparent border-b border-white/10 text-xs text-right text-cyber-accent w-16 focus:outline-none" />
                                                 )}
                                                 <div className="flex-1"></div>
                                                 <button onClick={() => toggleLayer(l.id)} className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400"><Trash2 size={12}/></button>
                                             </div>
                                         ))}
                                         <button onClick={addLayer} className="w-full py-2 border border-dashed border-white/20 rounded-lg text-xs text-gray-400 hover:text-white hover:border-white/40 transition-all flex items-center justify-center gap-2">
                                             <Plus size={12}/> ADD LAYER
                                         </button>
                                     </div>
                                 )}
                                 <div className="pt-4 border-t border-white/10 space-y-4">
                                     <RangeControl label="Learning Rate" value={modelConfig.learningRate || 0.001} min={0.0001} max={0.1} step={0.0001} onChange={v => setModelConfig({...modelConfig, learningRate: v})} />
                                     <RangeControl label="Dropout Rate" value={modelConfig.dropoutRate || 0.2} min={0} max={0.9} step={0.1} onChange={v => setModelConfig({...modelConfig, dropoutRate: v})} />
                                 </div>
                             </div>
                         )}

                         {activePanel === 'visuals' && (
                             <div className="space-y-6">
                                 <div className="space-y-4">
                                     <div className="flex items-center justify-between">
                                         <span className="text-xs text-gray-400">Environment Theme</span>
                                         <div className="flex gap-2">
                                             {['cyber', 'midnight'].map(t => (
                                                 <button key={t} onClick={() => setVisuals({...visuals, theme: t as any})} className={`w-6 h-6 rounded-full border ${visuals.theme === t ? 'border-cyber-accent bg-cyber-accent' : 'border-gray-600 bg-transparent'}`}></button>
                                             ))}
                                         </div>
                                     </div>
                                     <RangeControl label="Glow Intensity" value={visuals.glowIntensity} min={0} max={3} step={0.1} onChange={v => setVisuals({...visuals, glowIntensity: v})} />
                                     <RangeControl label="Particle Density" value={visuals.particleDensity} min={0} max={1} step={0.1} onChange={v => setVisuals({...visuals, particleDensity: v})} />
                                     <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                                         <span className="text-xs text-gray-300">Wireframe Mode</span>
                                         <button onClick={() => setVisuals({...visuals, wireframe: !visuals.wireframe})} className={`w-10 h-5 rounded-full relative transition-colors ${visuals.wireframe ? 'bg-cyber-accent' : 'bg-gray-700'}`}>
                                             <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${visuals.wireframe ? 'left-6' : 'left-1'}`}></div>
                                         </button>
                                     </div>
                                 </div>
                             </div>
                         )}
                         
                         {activePanel === 'code' && (
                             <div className="h-full flex flex-col">
                                 <div className="flex gap-2 mb-4">
                                     {['pytorch', 'tensorflow'].map(l => (
                                         <button key={l} onClick={() => setCodeLang(l as any)} className={`px-3 py-1 rounded text-[10px] font-bold uppercase transition-colors ${codeLang===l ? 'bg-cyber-accent text-black' : 'bg-white/5 text-gray-400'}`}>{l}</button>
                                     ))}
                                 </div>
                                 <div className="bg-black/50 p-4 rounded-lg border border-white/10 font-mono text-[10px] text-gray-300 overflow-x-auto whitespace-pre">
                                     {getGeneratedCode()}
                                 </div>
                             </div>
                         )}

                         {activePanel === 'export' && (
                             <div className="space-y-4">
                                 <p className="text-xs text-gray-400 leading-relaxed mb-4">
                                     Save your current neural architecture and visualization state. Use snapshots for presentations or export the raw 3D model for external CAD/Blender workflows.
                                 </p>
                                 <div className="grid grid-cols-2 gap-4">
                                     <button 
                                         onClick={() => threeSceneRef.current?.exportImage()}
                                         className="flex flex-col items-center justify-center p-6 bg-white/5 border border-white/10 rounded-xl hover:border-cyber-accent hover:bg-cyber-accent/5 transition-all group"
                                     >
                                         <Image size={32} className="text-gray-400 group-hover:text-cyber-accent mb-3 transition-colors"/>
                                         <span className="text-xs font-bold text-white uppercase tracking-wider">PNG Snapshot</span>
                                         <span className="text-[10px] text-gray-500 mt-1">High-Res Capture</span>
                                     </button>
                                     
                                     <button 
                                         onClick={() => threeSceneRef.current?.exportModel()}
                                         className="flex flex-col items-center justify-center p-6 bg-white/5 border border-white/10 rounded-xl hover:border-cyber-secondary hover:bg-cyber-secondary/5 transition-all group"
                                     >
                                         <Box size={32} className="text-gray-400 group-hover:text-cyber-secondary mb-3 transition-colors"/>
                                         <span className="text-xs font-bold text-white uppercase tracking-wider">GLTF Model</span>
                                         <span className="text-[10px] text-gray-500 mt-1">3D Binary Export</span>
                                     </button>
                                 </div>
                             </div>
                         )}
                     </div>
                 </div>
             )}

             {/* HOLOGRAPHIC CHAT */}
             {chatOpen && (
                 <div className="absolute right-8 bottom-28 w-96 h-[500px] z-40 glass-panel-pro rounded-2xl flex flex-col animate-slide-up shadow-[0_0_50px_rgba(0,0,0,0.5)] border border-cyber-secondary/30">
                     <div className="h-12 border-b border-white/5 flex items-center justify-between px-5 holo-gradient">
                         <div className="flex items-center gap-2">
                             <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                             <span className="text-xs font-bold text-white tracking-widest">GEMINI UPLINK</span>
                         </div>
                         <button onClick={() => setChatOpen(false)}><X size={16} className="text-white opacity-50 hover:opacity-100"/></button>
                     </div>
                     <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-black/20">
                         <div className="p-3 rounded-lg bg-white/5 border border-white/5 text-xs text-gray-300 font-mono">
                             System: Secure channel established. How can I assist with your architecture?
                         </div>
                         {messages.map(m => (
                             <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                 <div className={`max-w-[85%] p-3 rounded-xl text-xs leading-relaxed ${m.role === 'user' ? 'bg-cyber-accent text-black font-medium' : 'bg-white/10 text-gray-200 backdrop-blur-sm'}`}>
                                     {m.text}
                                 </div>
                             </div>
                         ))}
                         <div ref={messagesEndRef}></div>
                     </div>
                     <form className="p-4 border-t border-white/5 bg-black/20" onSubmit={(e) => { e.preventDefault(); if(chatInput.trim()) { setMessages([...messages, {id: Date.now().toString(), role: 'user', text: chatInput, timestamp: Date.now()}]); setChatInput(''); } }}>
                         <div className="relative">
                             <input className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-4 pr-10 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-cyber-secondary transition-all" placeholder="Enter query..." value={chatInput} onChange={e => setChatInput(e.target.value)} autoFocus />
                             <button type="submit" className="absolute right-2 top-2 p-1 text-cyber-secondary hover:text-white transition-colors"><Send size={16}/></button>
                         </div>
                     </form>
                 </div>
             )}
          </div>
      )}
    </div>
  );
};

export default App;
