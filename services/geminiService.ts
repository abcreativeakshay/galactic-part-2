
import { GoogleGenAI, Type } from "@google/genai";
import { MLModelConfig, StudioMode, LayerType, AlgorithmStep } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const MODEL_NAME = 'gemini-2.5-flash';

export const analyzeModel = async (config: MLModelConfig): Promise<string> => {
  try {
    let context = "";
    if (config.mode === StudioMode.NEURAL_ARCH) context = `Neural Network with layers: ${JSON.stringify(config.layers)}`;
    else if (config.mode === StudioMode.CLUSTERING) context = `K-Means Clustering with K=${config.k} on ${config.pointsCount} points`;
    else if (config.mode === StudioMode.DECISION_BOUNDARY) context = `SVM with ${config.kernel} kernel`;
    else if (config.mode === StudioMode.LOSS_LANDSCAPE) context = `Loss Landscape of type ${config.landscapeType}`;
    else if (config.mode === StudioMode.GENERAL_ML) {
        context = `Algorithm: ${config.algorithm}. Params: `;
        if (config.algorithm === 'DecisionTree') context += `Max Depth: ${config.treeDepth}`;
        if (config.algorithm === 'RandomForest') context += `Estimators: ${config.nEstimators}, Max Depth: ${config.treeDepth}`;
        if (config.algorithm === 'KNN') context += `Neighbors: ${config.nNeighbors}`;
    }
    else context = `ML Model Configuration: ${JSON.stringify(config)}`;

    const prompt = `
      You are a Senior Machine Learning Engineer. Analyze the following model configuration:
      ${context}
      
      Provide a concise, professional critique or insight suitable for visualization analysis. 
      Mention potential pitfalls (e.g., overfitting, curse of dimensionality, local minima).
      Keep it under 150 words.
    `;

    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: prompt,
      config: {
        systemInstruction: "You are an expert ML architect. Be technical but clear.",
        temperature: 0.7,
      }
    });

    return response.text || "Analysis failed.";
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    return "Unable to analyze model. Please check configuration.";
  }
};

export const suggestOptimization = async (currentConfig: MLModelConfig, goal: string): Promise<MLModelConfig | null> => {
  try {
    const prompt = `
      Current Config: ${JSON.stringify(currentConfig)}
      Optimization Goal: ${goal}
      
      Generate a new, optimized MLModelConfig JSON.
      Rules:
      1. Return ONLY the JSON object.
      2. Ensure parameters are valid for the mode ${currentConfig.mode}.
      3. For NEURAL_ARCH, use valid LayerTypes.
      4. For CLUSTERING, adjust K.
      5. For DECISION_BOUNDARY, adjust kernel/C.
      6. For GENERAL_ML, adjust algorithm or specific parameters (treeDepth, nEstimators, nNeighbors).
    `;

    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: prompt,
      config: {
        responseMimeType: "application/json" 
      }
    });

    const jsonText = response.text;
    if (!jsonText) return null;
    return JSON.parse(jsonText) as MLModelConfig;

  } catch (error) {
    console.error("Gemini Optimization Error:", error);
    return null;
  }
};

export const generateConfigFromModelName = async (modelName: string): Promise<MLModelConfig | null> => {
    try {
        const prompt = `
            The user wants to visualize the Machine Learning model: "${modelName}".
            
            Generate a valid JSON configuration object that best represents this model in a 3D visualization studio.
            
            Use the following TypeScript definitions as a guide for the JSON structure:
            
            enum StudioMode { 
                NEURAL_ARCH, CLUSTERING, DECISION_BOUNDARY, LOSS_LANDSCAPE, 
                GENERAL_ML, RL_ARENA, EMBEDDING_SPACE, 
                DIMENSIONALITY_REDUCTION, FEATURE_ENGINEERING, COMPUTER_VISION, 
                NLP, GENERATIVE_AI, AUDIO_PROCESSING, ENSEMBLE_METHODS, DATA_PREPROCESSING 
            }
            enum LayerType { INPUT, DENSE, CONV2D, POOLING, RNN, OUTPUT, ATTENTION }
            enum ActivationType { RELU, SIGMOID, TANH, SOFTMAX, LINEAR }
            
            interface MLModelConfig {
              mode: StudioMode; // Choose the best mode for the requested model
              
              // If mode is NEURAL_ARCH / CV / NLP / GEN_AI (e.g. ResNet, Transformer, MLP):
              layers?: { 
                 id: string; 
                 type: LayerType; 
                 neurons: number; 
                 activationFunction: ActivationType; 
                 name: string;
                 featureMapSize?: number; // Only for CONV2D (e.g. 4, 8, 16)
              }[];
              
              // If mode is CLUSTERING (e.g. K-Means, DBSCAN):
              k?: number; 
              
              // If mode is GENERAL_ML / ENSEMBLE (e.g. Random Forest, Decision Tree):
              algorithm?: 'DecisionTree' | 'RandomForest' | 'KNN';
              treeDepth?: number;
              nEstimators?: number;
              
              // If mode is DECISION_BOUNDARY (e.g. SVM):
              kernel?: 'linear' | 'rbf' | 'poly';
            }
            
            Constraint:
            - If it is a deep network (like ResNet, VGG, BERT, Diffusion), simplify it to 6-12 representative layers for visualization purposes.
            - Ensure strict JSON output.
        `;

        const response = await ai.models.generateContent({
            model: MODEL_NAME,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                temperature: 0.5
            }
        });

        const jsonText = response.text;
        if (!jsonText) return null;
        return JSON.parse(jsonText) as MLModelConfig;
    } catch (error) {
        console.error("Model Generation Error:", error);
        return null;
    }
};

export const generateSteps = async (config: MLModelConfig): Promise<AlgorithmStep[]> => {
    try {
        const prompt = `
            Based on the following ML configuration, generate 4-5 concise "Process Steps" describing how this algorithm works.
            
            Crucially, for each step, reference specific 3D visual elements the user would see in the visualization.
            
            Configuration: ${JSON.stringify(config)}
            
            Visual Elements Dictionary:
            - Neural Network: "Glowing spheres (neurons)", "Moving light pulses (signals)", "Layer connections"
            - Clustering: "Colored dots (data points)", "Large glowing spheres (centroids)", "Spatial grouping"
            - Loss Landscape: "3D terrain/surface", "Moving ball (optimizer)", "Valleys (minima)"
            - Decision Trees: "Branching nodes", "Cubes (leaves)"
            - Dimensionality Reduction: "Manifold warping", "Projection rays"
            - Diffusion/GenAI: "Noise clouds", "Denoising steps", "Image formation"
            
            Output strictly a JSON array of objects with keys: title, description, visualCue.
        `;
        
        const response = await ai.models.generateContent({
            model: MODEL_NAME,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                temperature: 0.5
            }
        });

        const jsonText = response.text;
        if (!jsonText) return [];
        return JSON.parse(jsonText) as AlgorithmStep[];
    } catch (e) {
        console.error("Step generation error", e);
        return [
            { title: "Initialization", description: "The system initializes parameters.", visualCue: "Static structures appear." },
            { title: "Processing", description: "The algorithm processes data.", visualCue: "Animations begin." }
        ];
    }
}

export const chatWithExpert = async (history: {role: string, parts: {text: string}[]}[], message: string) => {
    try {
        const chat = ai.chats.create({
            model: MODEL_NAME,
            history: history,
            config: {
                systemInstruction: "You are a helpful AI visualization assistant. You explain complex ML concepts (Deep Learning, Clustering, SVMs, RL, GenAI) to users exploring the 3D studio."
            }
        });
        
        const result = await chat.sendMessage({ message });
        return result.text;
    } catch (error) {
        console.error("Chat Error", error);
        throw error;
    }
}
