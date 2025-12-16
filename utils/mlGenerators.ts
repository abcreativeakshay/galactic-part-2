
import * as THREE from 'three';
import { DatasetType, OptimizerType } from '../types';

// --- Colors ---
const COLORS = {
  primary: new THREE.Color('#3b82f6'),   
  accent: new THREE.Color('#64ffda'),    
  secondary: new THREE.Color('#bd93f9'), 
  danger: new THREE.Color('#ef4444'),    
  success: new THREE.Color('#10b981'),   
  warning: new THREE.Color('#f59e0b')
};

// --- DATA GENERATORS ---

export const generateClusters = (k: number, count: number, noise: number = 1.0) => {
  const centers = [];
  for (let i = 0; i < k; i++) {
    centers.push({
      x: (Math.random() - 0.5) * 30,
      y: (Math.random() - 0.5) * 15,
      z: (Math.random() - 0.5) * 15,
      color: new THREE.Color().setHSL(i / k, 0.8, 0.6),
      id: `center_${i}`
    });
  }
  const points = [];
  for (let i = 0; i < count; i++) {
    const centerIdx = Math.floor(Math.random() * k);
    const center = centers[centerIdx];
    // Apply noise multiplier
    const spread = 5 * noise;
    points.push({
      x: center.x + (Math.random() - 0.5) * spread,
      y: center.y + (Math.random() - 0.5) * spread,
      z: center.z + (Math.random() - 0.5) * spread,
      color: center.color,
      id: `point_${i}_c${centerIdx}`
    });
  }
  return { centers, points };
};

export const generateClassificationData = (dataset: DatasetType = 'LINEAR', count: number = 400, noiseLevel: number = 1.0) => {
    const points = [];
    for (let i = 0; i < count; i++) {
        let x, y, z, label;
        // Base noise varies by dataset logic, multiply by noiseLevel
        const noise = (Math.random() - 0.5) * 1.5 * noiseLevel;

        if (dataset === 'LINEAR') {
            x = (Math.random() - 0.5) * 20; y = (Math.random() - 0.5) * 20; z = (Math.random() - 0.5) * 5;
            label = (x + y + noise) > 0 ? 1 : 0;
        } 
        else if (dataset === 'MOONS') {
            const isUpper = Math.random() > 0.5;
            label = isUpper ? 0 : 1;
            const t = Math.PI * Math.random(); 
            x = (isUpper ? 10 * Math.cos(t) - 5 : 10 * Math.cos(t) + 5) + noise;
            y = (isUpper ? 10 * Math.sin(t) + 2 : -10 * Math.sin(t) - 2) + noise;
            z = (Math.random() - 0.5) * 5;
        }
        else if (dataset === 'CIRCLES') {
            const angle = Math.random() * 2 * Math.PI;
            const isInner = Math.random() > 0.5;
            label = isInner ? 0 : 1;
            const r = isInner ? (Math.random() * 5) : (10 + Math.random() * 5);
            x = r * Math.cos(angle) + noise; y = r * Math.sin(angle) + noise; z = (Math.random() - 0.5) * 5;
        }
        else { // SPIRAL
             const t = Math.random() * 4 * Math.PI;
             const r = t * 1.5;
             const classOffset = Math.random() > 0.5 ? 0 : Math.PI;
             label = classOffset === 0 ? 0 : 1;
             x = r * Math.cos(t + classOffset) + noise; y = r * Math.sin(t + classOffset) + noise; z = (Math.random() - 0.5) * 5;
        }
        points.push({ x, y, z, color: label === 0 ? COLORS.primary : COLORS.secondary, id: `data_${i}`, label });
    }
    return points;
};

export const generateKNNData = (k: number, count: number, dataset: DatasetType, noise: number = 1.0) => {
   const points = generateClassificationData(dataset, count, noise);
   const queryPoint = { x: 0, y: 0, z: 0, id: 'query_target', color: COLORS.accent };
   const distances = points.map(p => ({ point: p, dist: Math.sqrt(p.x**2 + p.y**2 + p.z**2) }));
   distances.sort((a,b) => a.dist - b.dist);
   const neighbors = distances.slice(0, k);
   const links = neighbors.map(d => ({ start: queryPoint, end: d.point }));
   return { points, queryPoint, links, neighbors: neighbors.map(n => n.point) };
};

// --- LOSS LANDSCAPE & OPTIMIZERS ---

export const generateLossLandscapeGeometry = (type: string, resolution: number) => {
  const geometry = new THREE.PlaneGeometry(40, 40, resolution, resolution);
  const positions = geometry.attributes.position;
  
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i);
    const y = positions.getY(i); 
    let z = 0; 
    const nx = x / 5; const ny = y / 5;

    if (type === 'convex') z = (nx * nx + ny * ny) * 2;
    else if (type === 'saddle') z = (nx * nx - ny * ny) * 2;
    else if (type === 'rastrigin') z = (20 + (nx*nx - 10*Math.cos(2*Math.PI*nx)) + (ny*ny - 10*Math.cos(2*Math.PI*ny))); 
    else z = (10 * 2 + (nx * nx - 10 * Math.cos(2 * Math.PI * nx)) + (ny * ny - 10 * Math.cos(2 * Math.PI * ny))) * 0.5 - 10; // Non-convex default
    
    positions.setZ(i, z);
  }
  geometry.computeVertexNormals();
  return geometry;
};

// Simulates gradient descent path on the surface
export const generateOptimizerPath = (optimizer: OptimizerType, startX: number, startY: number, steps: number) => {
    const path = [];
    let x = startX;
    let y = startY;
    let vx = 0; // Velocity for Momentum/Adam
    let vy = 0;

    for (let i = 0; i < steps; i++) {
        // Calculate Height (Loss)
        const nx = x / 5; const ny = y / 5;
        const z = (10 * 2 + (nx * nx - 10 * Math.cos(2 * Math.PI * nx)) + (ny * ny - 10 * Math.cos(2 * Math.PI * ny))) * 0.5 - 10;
        
        path.push(new THREE.Vector3(x, y, z + 0.5)); // Lift slightly above surface

        // Calculate Gradient (approximate)
        const d = 0.1;
        const zx = ((10 * 2 + (((nx+d)* (nx+d)) - 10 * Math.cos(2 * Math.PI * (nx+d))) + (ny * ny - 10 * Math.cos(2 * Math.PI * ny))) * 0.5 - 10);
        const zy = ((10 * 2 + (nx * nx - 10 * Math.cos(2 * Math.PI * nx)) + (((ny+d)*(ny+d)) - 10 * Math.cos(2 * Math.PI * (ny+d)))) * 0.5 - 10);
        const gradX = (zx - z) / d;
        const gradY = (zy - z) / d;

        // Optimizer Logic
        if (optimizer === 'SGD') {
            const jitter = (Math.random() - 0.5) * 0.5; // Stochastic noise
            x -= (gradX * 0.1) + jitter;
            y -= (gradY * 0.1) + jitter;
        } else if (optimizer === 'Adam' || optimizer === 'Momentum') {
            vx = vx * 0.9 - gradX * 0.1;
            vy = vy * 0.9 - gradY * 0.1;
            x += vx;
            y += vy;
        } else {
            // Default Descent
            x -= gradX * 0.05;
            y -= gradY * 0.05;
        }

        // Bounds check
        if (Math.abs(x) > 18) x = Math.sign(x) * 18;
        if (Math.abs(y) > 18) y = Math.sign(y) * 18;
        
        // Check for convergence (basin)
        if (Math.sqrt(x*x + y*y) < 0.5) break; 
    }
    return path;
};

// --- TREES & FORESTS ---

const createTreeStructure = (maxDepth: number, rootX: number, rootY: number, rootZ: number, initialSpread: number, idPrefix: string) => {
  const nodes: any[] = [];
  const links: any[] = [];
  const createNode = (depth: number, x: number, y: number, z: number, spread: number): {x:number, y:number, z:number} => {
    const isLeaf = depth === maxDepth;
    const color = isLeaf ? (Math.random() > 0.5 ? COLORS.success : COLORS.danger) : COLORS.accent.clone().lerp(COLORS.primary, depth / maxDepth);
    const id = `${idPrefix}_d${depth}_${Math.floor(Math.random()*1000)}`;
    nodes.push({ x, y, z, id, depth, isLeaf, color });
    if (depth < maxDepth) {
      const nextSpread = spread * 0.55;
      const leftChild = createNode(depth + 1, x - spread, y - 2.5, z + (Math.random() * 2), nextSpread);
      links.push({ start: {x,y,z}, end: leftChild });
      const rightChild = createNode(depth + 1, x + spread, y - 2.5, z - (Math.random() * 2), nextSpread);
      links.push({ start: {x,y,z}, end: rightChild });
    }
    return {x,y,z};
  };
  createNode(0, rootX, rootY, rootZ, initialSpread);
  return { nodes, links };
};

export const generateTreeData = (maxDepth: number) => createTreeStructure(maxDepth, 0, 12, 0, 8, 'tree');

export const generateRandomForestData = (maxDepth: number, treeCount: number = 5) => {
  let allNodes: any[] = [];
  let allLinks: any[] = [];
  for (let i = 0; i < treeCount; i++) {
    const angle = (i / treeCount) * Math.PI * 2;
    const x = Math.cos(angle) * 15;
    const z = Math.sin(angle) * 15;
    const { nodes, links } = createTreeStructure(Math.max(2, maxDepth - 1), x, 10 + (Math.random()-0.5)*4, z, 5, `forest_${i}`);
    allNodes.push(...nodes);
    allLinks.push(...links);
  }
  return { nodes: allNodes, links: allLinks };
};
