
export enum LayerType {
  INPUT = 'INPUT',
  DENSE = 'DENSE',
  CONV2D = 'CONV2D',
  POOLING = 'POOLING',
  RNN = 'RNN',
  OUTPUT = 'OUTPUT',
  BATCH_NORM = 'BATCH_NORM',
  DROPOUT = 'DROPOUT',
  FLATTEN = 'FLATTEN'
}

export enum ActivationType {
  RELU = 'relu',
  SIGMOID = 'sigmoid',
  TANH = 'tanh',
  LEAKY_RELU = 'leaky_relu',
  SWISH = 'swish',
  LINEAR = 'linear',
  SOFTMAX = 'softmax',
  GELU = 'gelu'
}

export enum OptimizerType {
  SGD = 'SGD',
  ADAM = 'Adam',
  RMSPROP = 'RMSprop',
  ADAGRAD = 'Adagrad',
  MOMENTUM = 'Momentum'
}

export enum StudioMode {
  NEURAL_ARCH = 'NEURAL_ARCH',
  CLUSTERING = 'CLUSTERING',
  DECISION_BOUNDARY = 'DECISION_BOUNDARY',
  LOSS_LANDSCAPE = 'LOSS_LANDSCAPE',
  EMBEDDING_SPACE = 'EMBEDDING_SPACE',
  RL_ARENA = 'RL_ARENA',
  GENERAL_ML = 'GENERAL_ML',
  // New Categories
  DIMENSIONALITY_REDUCTION = 'DIMENSIONALITY_REDUCTION',
  FEATURE_ENGINEERING = 'FEATURE_ENGINEERING',
  COMPUTER_VISION = 'COMPUTER_VISION',
  NLP = 'NLP',
  GENERATIVE_AI = 'GENERATIVE_AI',
  AUDIO_PROCESSING = 'AUDIO_PROCESSING',
  ENSEMBLE_METHODS = 'ENSEMBLE_METHODS',
  DATA_PREPROCESSING = 'DATA_PREPROCESSING'
}

export type DatasetType = 'LINEAR' | 'MOONS' | 'CIRCLES' | 'SPIRAL' | 'RANDOM';

export interface VisualSettings {
  // Visual Engine
  theme: 'cyber' | 'midnight' | 'paper';
  showGrid: boolean;
  showStars: boolean;
  autoRotateSpeed: number;
  glowIntensity: number;
  particleDensity: number;
  signalSpeed: number;
  wireframe: boolean;
  connectionOpacity: number;
  
  // Specifics
  showClusterHulls: boolean;
  showCentroids: boolean;
  showSupportVectors: boolean;
  decisionSurfaceOpacity: number;
  lossWaterLevel: number;
  lossHeightScale: number;
  showContourLines: boolean;
  neuronSize: number;
}

export interface LayerConfig {
  id: string;
  type: LayerType;
  neurons: number; 
  activationFunction: ActivationType;
  name: string;
  kernelSize?: number;
  featureMapSize?: number; 
  stride?: number;
  padding?: boolean;
}

export interface MLModelConfig {
  mode: StudioMode;
  dataset?: DatasetType;
  noise?: number; 
  
  // Deep Learning
  layers?: LayerConfig[];
  learningRate?: number;
  dropoutRate?: number;
  optimizer?: OptimizerType;
  batchSize?: number;
  epochs?: number;
  lossFunction?: 'MSE' | 'CrossEntropy' | 'Hinge' | 'KLDivergence';
  connectionSparsity?: number; // 0 to 1

  // Clustering
  k?: number;
  pointsCount?: number;
  distanceMetric?: 'Euclidean' | 'Manhattan' | 'Cosine';
  clusteringAlgorithm?: 'KMeans' | 'DBSCAN' | 'Hierarchical';
  
  // SVM / Decision Boundary
  kernel?: 'linear' | 'rbf' | 'poly' | 'sigmoid';
  cParameter?: number; 
  gamma?: number; 
  
  // Loss Landscape
  landscapeType?: 'convex' | 'non-convex' | 'saddle' | 'rastrigin';
  resolution?: number;
  
  // General ML / Trees
  algorithm?: 'DecisionTree' | 'KNN' | 'RandomForest' | 'SVM';
  treeDepth?: number;
  nEstimators?: number;
  nNeighbors?: number;
  splitCriterion?: 'Gini' | 'Entropy' | 'LogLoss';
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export interface AlgorithmStep {
  title: string;
  description: string;
  visualCue: string;
}

export interface VizState {
  isPlaying: boolean;
  epoch: number;
  metric1: number;
  metric2: number;
  metric1Name: string;
  metric2Name: string;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  history1: number[];
  history2: number[];
  optimizerPath?: {x: number, y: number, z: number}[];
}

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
}

export interface AppSettings {
  highQuality: boolean;
  bloom: boolean;
  particles: boolean;
  gridOpacity: number;
  uiScale: number;
}
