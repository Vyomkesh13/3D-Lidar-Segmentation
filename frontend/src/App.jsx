import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import axios from 'axios';

const API_URL = '';

const CLASS_INFO = [
    { name: "barrier", color: [112,128,144], iou: 0.0 },
    { name: "bicycle", color: [220,20,60], iou: 0.2097 },
    { name: "bus", color: [255,127,80], iou: 0.9569 },
    { name: "car", color: [255,158,0], iou: 0.9468 },
    { name: "construction", color: [233,150,70], iou: 0.0 },
    { name: "motorcycle", color: [255,61,99], iou: 0.6658 },
    { name: "pedestrian", color: [0,0,230], iou: 0.7974 },
    { name: "traffic_cone", color: [47,79,79], iou: 0.0 },
    { name: "trailer", color: [255,140,0], iou: 0.0 },
    { name: "truck", color: [255,99,71], iou: 0.9024 },
    { name: "driveable", color: [0,207,191], iou: 0.8940 },
    { name: "other_flat", color: [175,0,75], iou: 0.0 },
    { name: "sidewalk", color: [75,0,75], iou: 0.3692 },
    { name: "terrain", color: [112,180,60], iou: 0.8038 },
    { name: "manmade", color: [222,184,135], iou: 0.8414 },
    { name: "vegetation", color: [0,175,0], iou: 0.9086 }
];

const TABS = ['3D View', 'Cameras', 'Metrics', 'Architecture'];

export default function App() {
    const mountRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const cameraRef = useRef(null);
    const controlsRef = useRef(null);
    const pointsRef = useRef(null);
    const coordsRef = useRef(null);
    const predictionsRef = useRef(null);
    const intensitiesRef = useRef(null);

    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState('Upload a LiDAR .bin file to visualize');
    const [stats, setStats] = useState(null);
    const [activeClasses, setActiveClasses] = useState(new Set(Array.from({length:16},(_,i)=>i)));
    const [colorMode, setColorMode] = useState('segmentation');
    const [pointSize, setPointSize] = useState(0.1);
    const [activeTab, setActiveTab] = useState('3D View');
    const [cameraImages, setCameraImages] = useState({});

    // ── Change 1: new state variables ──────────────────────────────────────
    const [samples, setSamples] = useState([]);
    const [selectedSample, setSelectedSample] = useState('');
    const [viewMode, setViewMode] = useState('prediction'); // prediction, groundtruth, confidence
    const [groundTruth, setGroundTruth] = useState(null);
    const [confidence, setConfidence] = useState(null);
    // ───────────────────────────────────────────────────────────────────────

    // Three.js scene setup
    useEffect(() => {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(60, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
        camera.position.set(0, 50, 50);
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        mountRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controlsRef.current = controls;

        const grid = new THREE.GridHelper(100, 50, 0x222222, 0x222222);
        scene.add(grid);

        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const handleResize = () => {
            if (!mountRef.current) return;
            camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        };
        window.addEventListener('resize', handleResize);
        return () => { window.removeEventListener('resize', handleResize); renderer.dispose(); };
    }, []);

    // ── Change 2: load samples on startup ──────────────────────────────────
    useEffect(() => {
        axios.get(`${API_URL}/samples`).then(r => setSamples(r.data.samples));
    }, []);
    // ───────────────────────────────────────────────────────────────────────

    const updatePointCloud = useCallback((coords, predictions, intensities, activeSet, mode, size) => {
        if (!coords || !sceneRef.current) return;
        if (pointsRef.current) sceneRef.current.remove(pointsRef.current);

        const filteredCoords = [];
        const filteredColors = [];

        coords.forEach((coord, i) => {
            const cls = predictions[i];
            if (!activeSet.has(cls)) return;
            filteredCoords.push(...coord);

            if (mode === 'segmentation') {
                const c = CLASS_INFO[cls].color;
                filteredColors.push(c[0]/255, c[1]/255, c[2]/255);
            } else if (mode === 'height') {
                const z = coord[2];
                const t = Math.min(Math.max((z + 2) / 6, 0), 1);
                // Blue(low) → Green → Yellow → Red(high)
                const r = Math.min(1, Math.max(0, t * 2 - 0.5));
                const g = Math.min(1, Math.max(0, 1 - Math.abs(t * 2 - 1)));
                const b = Math.min(1, Math.max(0, 1 - t * 2));
                filteredColors.push(r, g, b);
            } else if (mode === 'intensity') {
                const rawVal = intensities ? intensities[i] : 0.5;
                const val = Math.min(Math.max(rawVal * 3.0, 0), 1);
                const r = Math.min(1, val * 3);
                const g = Math.min(1, Math.max(0, val * 3 - 1));
                const b = Math.min(1, Math.max(0, val * 3 - 2));
                filteredColors.push(r, g, b);
            }
        });

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(filteredCoords), 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(filteredColors), 3));

        const material = new THREE.PointsMaterial({ size, vertexColors: true });
        const points = new THREE.Points(geometry, material);
        sceneRef.current.add(points);
        pointsRef.current = points;
    }, []);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setLoading(true);
        setStatus('Running segmentation...');

        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await axios.post(`${API_URL}/segment`, formData);
            const data = response.data;

            coordsRef.current = data.coords;
            predictionsRef.current = data.predictions;
            intensitiesRef.current = data.intensities;

            updatePointCloud(data.coords, data.predictions, data.intensities, activeClasses, colorMode, pointSize);

            const classCounts = new Array(16).fill(0);
            data.predictions.forEach(p => classCounts[p]++);
            setStats({ numPoints: data.num_points, classCounts });
            setCameraImages(data.camera_images || {});
            setStatus(`✅ Segmented ${data.num_points.toLocaleString()} points`);
        } catch (err) {
            setStatus(`❌ Error: ${err.message}`);
        }
        setLoading(false);
    };

    // ── Change 3: sample selector handler ──────────────────────────────────
    const handleSampleSelect = async (token) => {
        if (!token) return;
        setSelectedSample(token);
        setLoading(true);
        setStatus('Running segmentation...');
        try {
            const response = await axios.get(`${API_URL}/segment_by_token/${token}`);
            const data = response.data;
            coordsRef.current = data.coords;
            predictionsRef.current = data.predictions;
            intensitiesRef.current = data.intensities;
            setGroundTruth(data.ground_truth);
            setConfidence(data.confidence);
            console.log('groundTruth length:', data.ground_truth?.length);
            console.log('confidence length:', data.confidence?.length);
            console.log('confidence sample:', data.confidence?.slice(0,5));
            updatePointCloud(data.coords, data.predictions, data.intensities, activeClasses, colorMode, pointSize);
            const classCounts = new Array(16).fill(0);
            data.predictions.forEach(p => classCounts[p]++);
            setStats({ numPoints: data.num_points, classCounts });
            setCameraImages(data.camera_images || {});
            setStatus(`✅ Segmented ${data.num_points.toLocaleString()} points`);
        } catch (err) {
            setStatus(`❌ Error: ${err.message}`);
        }
        setLoading(false);
    };
    // ───────────────────────────────────────────────────────────────────────

    const toggleClass = (i) => {
        const newSet = new Set(activeClasses);
        if (newSet.has(i)) newSet.delete(i); else newSet.add(i);
        setActiveClasses(newSet);
        if (coordsRef.current) updatePointCloud(coordsRef.current, predictionsRef.current, intensitiesRef.current, newSet, colorMode, pointSize);
    };

    const handleColorMode = (mode) => {
        setColorMode(mode);
        if (coordsRef.current) updatePointCloud(coordsRef.current, predictionsRef.current, intensitiesRef.current, activeClasses, mode, pointSize);
    };

    const handlePointSize = (size) => {
        setPointSize(size);
        if (pointsRef.current) pointsRef.current.material.size = size;
    };

    const selectAll = () => {
        const newSet = new Set(Array.from({length:16},(_,i)=>i));
        setActiveClasses(newSet);
        if (coordsRef.current) updatePointCloud(coordsRef.current, predictionsRef.current, intensitiesRef.current, newSet, colorMode, pointSize);
    };

    const selectNone = () => {
        const newSet = new Set();
        setActiveClasses(newSet);
        if (pointsRef.current) sceneRef.current.remove(pointsRef.current);
    };

    const renderCamerasTab = () => (
        <div style={{ padding: 16 }}>
            <h2 style={{ fontSize: 14, color: '#00d4ff', marginBottom: 12 }}>📷 6 Camera Views (DINOv2 Input)</h2>
            {Object.keys(cameraImages).length === 0 ? (
                <p style={{ color: '#555', fontSize: 13 }}>Upload a .bin file to see camera images</p>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {Object.entries(cameraImages).map(([cam, b64]) => (
                        <div key={cam}>
                            <p style={{ fontSize: 10, color: '#666', marginBottom: 3 }}>{cam}</p>
                            <img src={`data:image/jpeg;base64,${b64}`} style={{ width: '100%', borderRadius: 4, border: '1px solid #222' }} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );

    const renderMetricsTab = () => (
        <div style={{ padding: 16 }}>
            <h2 style={{ fontSize: 14, color: '#00d4ff', marginBottom: 4 }}>📊 Model Performance</h2>
            <p style={{ fontSize: 11, color: '#555', marginBottom: 16 }}>PTv3 + DINOv2 Fusion — Innovation 1 (25 epochs + CosineAnnealingLR)</p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 16 }}>
                {[
                    { label: 'mIoU', value: '0.6913', color: '#00d4ff' },
                    { label: 'allAcc', value: '92.06%', color: '#00ff88' },
                    { label: 'mAcc', value: '73.82%', color: '#ff8800' },
                ].map(m => (
                    <div key={m.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 10, textAlign: 'center' }}>
                        <p style={{ fontSize: 18, fontWeight: 700, color: m.color }}>{m.value}</p>
                        <p style={{ fontSize: 11, color: '#555' }}>{m.label}</p>
                    </div>
                ))}
            </div>

            <h3 style={{ fontSize: 12, color: '#aaa', marginBottom: 8 }}>Per-Class IoU</h3>
            {CLASS_INFO.map((cls, i) => (
                <div key={i} style={{ marginBottom: 6 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{ width: 8, height: 8, borderRadius: 2, background: `rgb(${cls.color.join(',')})` }} />
                            <span style={{ fontSize: 11, color: '#ccc' }}>{cls.name}</span>
                        </div>
                        <span style={{ fontSize: 11, color: '#aaa' }}>{(cls.iou * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 5, background: '#222', borderRadius: 3 }}>
                        <div style={{ height: '100%', width: `${cls.iou * 100}%`, background: `rgb(${cls.color.join(',')})`, borderRadius: 3 }} />
                    </div>
                </div>
            ))}

            <div style={{ marginTop: 16, padding: 10, background: '#1a1a1a', borderRadius: 6 }}>
                <h3 style={{ fontSize: 12, color: '#aaa', marginBottom: 8 }}>Model Comparison</h3>
                {[
                    { name: 'PTv3 only', miou: 0.65, color: '#555' },
                    { name: 'DirectFusion', miou: 0.6768, color: '#888' },
                    { name: 'ProjectionFusion (10ep)', miou: 0.6896, color: '#aaa' },
                    { name: '25ep + CosineAnnealingLR', miou: 0.6913, color: '#00d4ff' },
                    
                ].map(m => (
                    <div key={m.name} style={{ marginBottom: 6 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span style={{ fontSize: 11, color: m.color }}>{m.name}</span>
                            <span style={{ fontSize: 11, color: m.color }}>{(m.miou * 100).toFixed(2)}%</span>
                        </div>
                        <div style={{ height: 5, background: '#222', borderRadius: 3 }}>
                            <div style={{ height: '100%', width: `${(m.miou - 0.6) * 500}%`, background: m.color, borderRadius: 3 }} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );

    const renderArchitectureTab = () => (
        <div style={{ padding: 16 }}>
            <h2 style={{ fontSize: 14, color: '#00d4ff', marginBottom: 16 }}>🏗 Model Architecture</h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                    { label: 'LiDAR Input', desc: 'Raw point cloud (x,y,z,intensity)', color: '#333', icon: '📡' },
                    { label: 'PTv3 Backbone', desc: 'Point Transformer V3 (46.16M params)', color: '#1a3a5c', icon: '🔷' },
                    { label: 'PTv3 Features', desc: '64-dim per voxel features', color: '#1a3a5c', icon: '⬇' },
                    { label: 'Camera Input', desc: '6× nuScenes cameras (1600×900)', color: '#333', icon: '📷' },
                    { label: 'DINOv2 Backbone', desc: 'ViT-Small/14 (22.06M params)', color: '#3a1a5c', icon: '🔮' },
                    { label: 'DINOv2 Features', desc: '384-dim per patch features', color: '#3a1a5c', icon: '⬇' },
                    { label: 'Point→Image Projection', desc: 'Project LiDAR points onto camera pixels', color: '#1a4a3a', icon: '🎯' },
                    { label: 'DINOPT Fusion', desc: 'Concat PTv3 + DINOv2 features → MLP', color: '#4a3a1a', icon: '⚡' },
                    { label: 'Segmentation Head', desc: 'Linear(128 → 16 classes)', color: '#4a1a1a', icon: '🎨' },
                    { label: 'Output', desc: '16-class semantic segmentation', color: '#1a4a1a', icon: '✅' },
                ].map((item, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', background: item.color, borderRadius: 6, border: '1px solid #333' }}>
                        <span style={{ fontSize: 16 }}>{item.icon}</span>
                        <div>
                            <p style={{ fontSize: 12, fontWeight: 600, color: '#fff' }}>{item.label}</p>
                            <p style={{ fontSize: 10, color: '#aaa' }}>{item.desc}</p>
                        </div>
                    </div>
                ))}
            </div>

            <div style={{ marginTop: 16, padding: 10, background: '#1a1a1a', borderRadius: 6 }}>
                <p style={{ fontSize: 11, color: '#555' }}>Training: AdamW lr=0.002, CosineAnnealingLR</p>
                <p style={{ fontSize: 11, color: '#555' }}>Epochs: 25 | Batch: 12 | Loss: CE + Lovasz</p>
                <p style={{ fontSize: 11, color: '#555' }}>Dataset: nuScenes mini (404 samples)</p>
            </div>
        </div>
    );
    const renderLegend = () => {
        if (!stats) return null;
        
        if (colorMode === 'segmentation' || viewMode === 'groundtruth') {
            return (
                <div style={{ position:'absolute', bottom:40, left:16, background:'rgba(0,0,0,0.7)', borderRadius:6, padding:'8px 12px', pointerEvents:'none' }}>
                    <p style={{ fontSize:10, color:'#aaa', marginBottom:4 }}>Class Legend</p>
                    {CLASS_INFO.map((cls, i) => {
                        if (!stats.classCounts[i] || !activeClasses.has(i)) return null;
                        return (
                            <div key={i} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:2 }}>
                                <div style={{ width:8, height:8, borderRadius:2, background:`rgb(${cls.color.join(',')})`, flexShrink:0 }} />
                                <span style={{ fontSize:10, color:'#ccc' }}>{cls.name}</span>
                                <span style={{ fontSize:9, color:'#555' }}>{(stats.classCounts[i]/stats.numPoints*100).toFixed(1)}%</span>
                            </div>
                        );
                    })}
                </div>
            );
        }
    
        if (colorMode === 'height') {
            return (
                <div style={{ position:'absolute', bottom:40, left:16, background:'rgba(0,0,0,0.7)', borderRadius:6, padding:'8px 12px', pointerEvents:'none' }}>
                    <p style={{ fontSize:10, color:'#aaa', marginBottom:6 }}>Height (Z)</p>
                    {[['High (+4m)', '#ff0000'], ['Mid (+0m)', '#00ff00'], ['Low (-2m)', '#0000ff']].map(([label, color]) => (
                        <div key={label} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:3 }}>
                            <div style={{ width:8, height:8, borderRadius:2, background:color }} />
                            <span style={{ fontSize:10, color:'#ccc' }}>{label}</span>
                        </div>
                    ))}
                </div>
            );
        }
    
        if (colorMode === 'intensity') {
            return (
                <div style={{ position:'absolute', bottom:40, left:16, background:'rgba(0,0,0,0.7)', borderRadius:6, padding:'8px 12px', pointerEvents:'none' }}>
                    <p style={{ fontSize:10, color:'#aaa', marginBottom:6 }}>LiDAR Intensity</p>
                    {[['High (bright surface)', '#ff4400'], ['Medium', '#ffaa00'], ['Low (dark surface)', '#000066']].map(([label, color]) => (
                        <div key={label} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:3 }}>
                            <div style={{ width:8, height:8, borderRadius:2, background:color }} />
                            <span style={{ fontSize:10, color:'#ccc' }}>{label}</span>
                        </div>
                    ))}
                </div>
            );
        }
    
        if (viewMode === 'confidence') {
            return (
                <div style={{ position:'absolute', bottom:40, left:16, background:'rgba(0,0,0,0.7)', borderRadius:6, padding:'8px 12px', pointerEvents:'none' }}>
                    <p style={{ fontSize:10, color:'#aaa', marginBottom:6 }}>Model Confidence</p>
                    {[['Very confident (>99%)', '#ff8800'], ['Confident (95-99%)', '#aa5500'], ['Uncertain (<95%)', '#221100']].map(([label, color]) => (
                        <div key={label} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:3 }}>
                            <div style={{ width:8, height:8, borderRadius:2, background:color }} />
                            <span style={{ fontSize:10, color:'#ccc' }}>{label}</span>
                        </div>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <div style={{ display:'flex', height:'100vh', background:'#0a0a0a', color:'#fff', fontFamily:'Segoe UI,sans-serif' }}>
            {/* Sidebar */}
            <div style={{ width:300, background:'#111', display:'flex', flexDirection:'column', borderRight:'1px solid #222' }}>
                
                {/* Header */}
                <div style={{ padding:'12px 16px', borderBottom:'1px solid #222' }}>
                    <h1 style={{ fontSize:15, fontWeight:700, color:'#00d4ff' }}>3D LiDAR Segmentation</h1>
                    <p style={{ fontSize:10, color:'#555', marginTop:2 }}>PTv3 + DINOv2 Fusion | mIoU: 0.6913</p>
                </div>

                {/* Tabs */}
                <div style={{ display:'flex', borderBottom:'1px solid #222' }}>
                    {TABS.map(tab => (
                        <button key={tab} onClick={() => setActiveTab(tab)} style={{
                            flex:1, padding:'8px 0', fontSize:10, border:'none', cursor:'pointer',
                            background: activeTab===tab ? '#1a1a1a' : 'transparent',
                            color: activeTab===tab ? '#00d4ff' : '#555',
                            borderBottom: activeTab===tab ? '2px solid #00d4ff' : '2px solid transparent'
                        }}>{tab}</button>
                    ))}
                </div>

                {/* Tab Content */}
                <div style={{ flex:1, overflowY:'auto' }}>
                    {activeTab === '3D View' && (
                        <div style={{ padding:16, display:'flex', flexDirection:'column', gap:12 }}>

                            {/* ── Change 5: sample selector at top ───────────────── */}
                            {samples.length > 0 && (
                                <div>
                                    <p style={{ fontSize:12, fontWeight:600, color:'#aaa', marginBottom:6 }}>Quick Sample</p>
                                    <select onChange={e => handleSampleSelect(e.target.value)} value={selectedSample}
                                        style={{ width:'100%', padding:'6px 8px', background:'#222', color:'#ccc', border:'1px solid #333', borderRadius:4, fontSize:11 }}>
                                        <option value="">Select a sample...</option>
                                        {samples.map(s => (
                                            <option key={s.token} value={s.token}>{s.split}: {s.token.slice(0,12)}...</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                            {/* ────────────────────────────────────────────────────── */}

                            <label style={{ display:'block', padding:'8px 12px', background:'#00d4ff', color:'#000', borderRadius:6, textAlign:'center', cursor:'pointer', fontWeight:600, fontSize:13 }}>
                                {loading ? '⏳ Processing...' : '📂 Upload .bin File'}
                                <input type="file" accept=".bin,.pcd.bin" onChange={handleFileUpload} style={{ display:'none' }} disabled={loading} />
                            </label>
                            <p style={{ fontSize:11, color:'#888', textAlign:'center', marginTop:-8 }}>{status}</p>

                            <div>
                                <p style={{ fontSize:12, fontWeight:600, color:'#aaa', marginBottom:6 }}>Color Mode</p>
                                <div style={{ display:'flex', gap:4 }}>
                                    {['segmentation','height','intensity'].map(mode => (
                                        <button key={mode} onClick={() => handleColorMode(mode)} style={{
                                            flex:1, padding:'4px 0', fontSize:10, borderRadius:4, border:'none', cursor:'pointer',
                                            background: colorMode===mode ? '#00d4ff' : '#222', color: colorMode===mode ? '#000' : '#aaa'
                                        }}>{mode}</button>
                                    ))}
                                </div>
                            </div>

                            {/* ── Change 4: view mode selector after color mode ───── */}
                            <div>
                                <p style={{ fontSize:12, fontWeight:600, color:'#aaa', marginBottom:6 }}>View Mode</p>
                                <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
                                    {['prediction','groundtruth','confidence'].map(mode => (
                                        <button key={mode} onClick={() => {
                                            setViewMode(mode);
                                            if (mode === 'groundtruth') {
                                                const gt = groundTruth || predictionsRef.current;
                                                updatePointCloud(coordsRef.current, gt, intensitiesRef.current, activeClasses, colorMode, pointSize);
                                            } else if (mode === 'confidence' && confidence) {
                                                if (pointsRef.current) sceneRef.current.remove(pointsRef.current);
                                                const geometry = new THREE.BufferGeometry();
                                                const positions = new Float32Array(coordsRef.current.flat());
                                                const colors = new Float32Array(coordsRef.current.length * 3);
                                                confidence.forEach((c, i) => { 
                                                    const stretched = Math.pow(c, 8);
                                                    colors[i*3]=stretched; 
                                                    colors[i*3+1]=stretched*0.3; 
                                                    colors[i*3+2]=0; 
                                                });
                                                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                                                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                                                const pts = new THREE.Points(geometry, new THREE.PointsMaterial({ size: pointSize, vertexColors: true }));
                                                sceneRef.current.add(pts);
                                                pointsRef.current = pts;
                                            } else {
                                                updatePointCloud(coordsRef.current, predictionsRef.current, intensitiesRef.current, activeClasses, colorMode, pointSize);
                                            }
                                        }} style={{
                                            flex:1, padding:'4px 0', fontSize:10, borderRadius:4, border:'none', cursor:'pointer',
                                            background: viewMode===mode ? '#00d4ff' : '#222', color: viewMode===mode ? '#000' : '#aaa'
                                        }}>{mode}</button>
                                    ))}
                                </div>
                            </div>
                            {/* ────────────────────────────────────────────────────── */}

                            <div>
                                <p style={{ fontSize:12, fontWeight:600, color:'#aaa', marginBottom:4 }}>Point Size: {pointSize.toFixed(2)}</p>
                                <input type="range" min="0.05" max="0.5" step="0.01" value={pointSize}
                                    onChange={e => handlePointSize(parseFloat(e.target.value))}
                                    style={{ width:'100%', accentColor:'#00d4ff' }} />
                            </div>

                            {stats && (
                                <div style={{ background:'#1a1a1a', borderRadius:6, padding:10 }}>
                                    <p style={{ fontSize:12, color:'#aaa', marginBottom:6 }}>
                                        Total: <b style={{ color:'#fff' }}>{CLASS_INFO.reduce((s,_,i) => s + (activeClasses.has(i) ? stats.classCounts[i] : 0), 0).toLocaleString()}</b> points
                                    </p>
                                    {CLASS_INFO.map((cls, i) => {
                                        const count = activeClasses.has(i) ? stats.classCounts[i] : 0;
                                        if (count === 0) return null;
                                        const pct = (count / stats.numPoints * 100).toFixed(1);
                                        return (
                                            <div key={i} style={{ marginBottom:3 }}>
                                                <div style={{ display:'flex', justifyContent:'space-between', marginBottom:1 }}>
                                                    <span style={{ fontSize:10, color:'#aaa' }}>{cls.name}</span>
                                                    <span style={{ fontSize:10, color:'#666' }}>{pct}%</span>
                                                </div>
                                                <div style={{ height:4, background:'#222', borderRadius:2 }}>
                                                    <div style={{ height:'100%', width:`${pct}%`, background:`rgb(${cls.color.join(',')})`, borderRadius:2 }} />
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}

                            <div>
                                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:6 }}>
                                    <p style={{ fontSize:12, fontWeight:600, color:'#aaa' }}>Classes</p>
                                    <div style={{ display:'flex', gap:4 }}>
                                        <button onClick={selectAll} style={{ fontSize:10, padding:'2px 6px', background:'#222', color:'#aaa', border:'none', borderRadius:3, cursor:'pointer' }}>All</button>
                                        <button onClick={selectNone} style={{ fontSize:10, padding:'2px 6px', background:'#222', color:'#aaa', border:'none', borderRadius:3, cursor:'pointer' }}>None</button>
                                    </div>
                                </div>
                                {CLASS_INFO.map((cls, i) => (
                                    <div key={i} onClick={() => toggleClass(i)} style={{
                                        display:'flex', alignItems:'center', gap:8, marginBottom:4,
                                        cursor:'pointer', opacity: activeClasses.has(i) ? 1 : 0.3,
                                        padding:'3px 6px', borderRadius:4,
                                        background: activeClasses.has(i) ? '#1a1a1a' : 'transparent'
                                    }}>
                                        <div style={{ width:10, height:10, borderRadius:2, background:`rgb(${cls.color.join(',')})`, flexShrink:0 }} />
                                        <span style={{ fontSize:11, color:'#ccc', flex:1 }}>{cls.name}</span>
                                        {stats && activeClasses.has(i) && <span style={{ fontSize:10, color:'#555' }}>{stats.classCounts[i]}</span>}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    {activeTab === 'Cameras' && renderCamerasTab()}
                    {activeTab === 'Metrics' && renderMetricsTab()}
                    {activeTab === 'Architecture' && renderArchitectureTab()}
                </div>

                {/* Footer */}
                <div style={{ padding:10, borderTop:'1px solid #222', fontSize:11 }}>
                    <p style={{ color:'#555' }}>PTv3 (46M) + DINOv2 (22M) | nuScenes mini</p>
                    <p style={{ color:'#00d4ff', fontWeight:600 }}>mIoU: 0.6913 | allAcc: 92.06%</p>
                </div>
            </div>

            {/* 3D Viewer */}
            <div ref={mountRef} style={{ flex:1, position:'relative' }}>
                {!stats && (
                    <div style={{ position:'absolute', top:'50%', left:'50%', transform:'translate(-50%,-50%)', textAlign:'center', pointerEvents:'none' }}>
                        <p style={{ fontSize:48, marginBottom:16 }}>🎯</p>
                        <p style={{ color:'#444', fontSize:16 }}>Upload a nuScenes LiDAR .bin file</p>
                        <p style={{ color:'#333', fontSize:13, marginTop:8 }}>to visualize 3D point cloud segmentation</p>
                    </div>
                )}
                {renderLegend()}
                <div style={{ position:'absolute', bottom:16, right:16, fontSize:11, color:'#333', textAlign:'right', pointerEvents:'none' }}>
                    <p>🖱 Left drag: rotate | Right drag: pan | Scroll: zoom</p>
                </div>
            </div>
        </div>
    );
}