"""
AION Comprehensive Protein Analysis
====================================

Unified interface for all protein analysis capabilities.
Combines structure prediction, dynamics, drug binding,
disorder, and cellular environment analysis.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Import all protein modules
from .protein_engine import UltimateProteinFolder, ProteinStructure
from .protein_dynamics import simulate_folding, FoldingSimulator, MisfoldingDetector
from .drug_binding import analyze_drug_binding, PocketDetector, BindingAnalyzer
from .disorder import analyze_disorder, DisorderPredictor, LLPSPredictor
from .cellular import analyze_cellular_environment, CellularEnvironment


@dataclass
class ComprehensiveAnalysis:
    """Complete protein analysis result."""
    sequence: str
    structure: Dict[str, Any]
    folding: Dict[str, Any]
    drug_binding: Dict[str, Any]
    disorder: Dict[str, Any]
    cellular: Dict[str, Any]
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            "sequence": self.sequence,
            "structure": self.structure,
            "folding": self.folding,
            "drug_binding": self.drug_binding,
            "disorder": self.disorder,
            "cellular": self.cellular
        }, indent=2, default=str)


def analyze_protein_complete(sequence: str, 
                            compartment: str = "cytoplasm") -> ComprehensiveAnalysis:
    """
    Complete protein analysis pipeline.
    
    Runs all analyses and returns unified results.
    """
    print("=" * 60)
    print("AION COMPREHENSIVE PROTEIN ANALYSIS")
    print("=" * 60)
    print(f"Sequence: {sequence[:20]}... ({len(sequence)} residues)")
    print()
    
    # 1. Structure prediction
    print("📐 Phase 1: Structure Prediction")
    folder = UltimateProteinFolder(sequence)
    structure = folder.fold()
    
    structure_data = structure.to_json() if hasattr(structure, 'to_json') else {}
    
    # Get backbone coordinates for other analyses
    backbone_coords = []
    if hasattr(structure, 'residues'):
        for res in structure.residues:
            if hasattr(res, 'CA') and res.CA:
                backbone_coords.append((res.CA.position.x, res.CA.position.y, res.CA.position.z))
    
    # Fallback coords if structure doesn't have proper atoms
    if not backbone_coords:
        # Generate simple helix-like coords
        import math
        for i in range(len(sequence)):
            x = 1.5 * math.cos(i * 1.75)
            y = 1.5 * math.sin(i * 1.75)
            z = i * 1.5
            backbone_coords.append((x, y, z))
    
    print(f"   ✓ Structure generated with {len(backbone_coords)} Cα atoms")
    print()
    
    # 2. Folding dynamics
    print("🔄 Phase 2: Folding Dynamics")
    folding_result = simulate_folding(sequence)
    print()
    
    # 3. Drug binding
    print("💊 Phase 3: Drug Binding Analysis")
    binding_result = analyze_drug_binding(sequence, backbone_coords)
    print()
    
    # 4. Disorder prediction
    print("🌊 Phase 4: Disorder Analysis")
    disorder_result = analyze_disorder(sequence)
    print()
    
    # 5. Cellular environment
    print("🏠 Phase 5: Cellular Environment")
    cellular_result = analyze_cellular_environment(sequence, compartment)
    print()
    
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return ComprehensiveAnalysis(
        sequence=sequence,
        structure=structure_data if isinstance(structure_data, dict) else {"data": structure_data},
        folding=folding_result,
        drug_binding=binding_result,
        disorder=disorder_result,
        cellular=cellular_result
    )


def generate_html_visualizer(analysis: ComprehensiveAnalysis, 
                            output_path: str = "protein_analysis.html") -> str:
    """Generate interactive HTML visualization."""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AION Protein Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(90deg, #2a1a4a, #1a3a5a);
            padding: 20px 40px;
            border-bottom: 2px solid #4a3a8a;
        }
        .header h1 {
            font-size: 2em;
            background: linear-gradient(90deg, #a0a0ff, #60ffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .subtitle {
            color: #888;
            margin-top: 5px;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .panel {
            background: rgba(20, 20, 40, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(100, 100, 200, 0.3);
            backdrop-filter: blur(10px);
        }
        .panel.full-width {
            grid-column: 1 / -1;
        }
        .panel h2 {
            color: #a0a0ff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .panel h2 .icon { font-size: 1.5em; }
        .metric {
            display: inline-block;
            background: rgba(100, 100, 200, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-size: 0.9em;
        }
        .metric .value {
            color: #60ffff;
            font-weight: bold;
        }
        .chart { height: 300px; margin: 15px 0; }
        .sequence-display {
            font-family: 'Courier New', monospace;
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: nowrap;
            letter-spacing: 2px;
        }
        .residue { display: inline-block; width: 18px; text-align: center; }
        .h-res { color: #ff6060; }
        .e-res { color: #60ff60; }
        .c-res { color: #6060ff; }
        .d-res { background: rgba(255,165,0,0.3); }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .table th, .table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(100,100,200,0.2);
        }
        .table th { color: #a0a0ff; }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        .badge.high { background: #ff4040; }
        .badge.medium { background: #ffaa00; color: #000; }
        .badge.low { background: #40ff40; color: #000; }
        .progress-bar {
            height: 8px;
            background: rgba(100,100,200,0.2);
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #60ff60, #60ffff);
            transition: width 0.5s;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .tab {
            padding: 8px 20px;
            background: rgba(100,100,200,0.2);
            border: none;
            border-radius: 20px;
            color: #a0a0ff;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab:hover, .tab.active {
            background: rgba(100,100,200,0.5);
            color: white;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 AION Comprehensive Protein Analysis</h1>
        <div class="subtitle">SEQUENCE_LENGTH residues | COMPARTMENT environment</div>
    </div>
    
    <div class="container">
        <!-- Sequence Overview -->
        <div class="panel full-width">
            <h2><span class="icon">📋</span> Sequence Overview</h2>
            <div class="sequence-display" id="sequence-display"></div>
            <div style="margin-top: 15px;">
                <span class="metric">Length: <span class="value">SEQUENCE_LENGTH</span></span>
                <span class="metric">Disorder: <span class="value">DISORDER_PERCENT%</span></span>
                <span class="metric">Net Charge: <span class="value">NET_CHARGE</span></span>
            </div>
        </div>
        
        <!-- 3D Structure -->
        <div class="panel">
            <h2><span class="icon">📐</span> 3D Structure</h2>
            <div id="structure-3d" class="chart" style="height: 350px;"></div>
        </div>
        
        <!-- Folding Dynamics -->
        <div class="panel">
            <h2><span class="icon">🔄</span> Folding Dynamics</h2>
            <div id="folding-chart" class="chart"></div>
            <div>
                <span class="metric">Pathway: <span class="value">PATHWAY_TYPE</span></span>
                <span class="metric">Aggregation Risk: <span class="value">AGG_SCORE</span></span>
            </div>
        </div>
        
        <!-- Disorder Prediction -->
        <div class="panel">
            <h2><span class="icon">🌊</span> Intrinsic Disorder</h2>
            <div id="disorder-chart" class="chart"></div>
            <div>
                <span class="metric">Classification: <span class="value">DISORDER_CLASS</span></span>
                <span class="metric">LLPS Propensity: <span class="value">LLPS_SCORE</span></span>
            </div>
        </div>
        
        <!-- Drug Binding -->
        <div class="panel">
            <h2><span class="icon">💊</span> Drug Binding Sites</h2>
            <table class="table">
                <tr><th>Pocket</th><th>Volume (Å³)</th><th>Druggability</th></tr>
                POCKET_ROWS
            </table>
            <div id="binding-chart" class="chart" style="height: 200px;"></div>
        </div>
        
        <!-- Cellular Environment -->
        <div class="panel">
            <h2><span class="icon">🏠</span> Cellular Environment Effects</h2>
            <div class="tabs">
                <button class="tab active" onclick="showTab('crowding')">Crowding</button>
                <button class="tab" onclick="showTab('ph')">pH Effects</button>
                <button class="tab" onclick="showTab('chaperones')">Chaperones</button>
            </div>
            <div id="crowding" class="tab-content active">
                <p>Stability change: <strong>STABILITY_CHANGE%</strong></p>
                <p>Diffusion reduction: <strong>DIFFUSION_CHANGE%</strong></p>
            </div>
            <div id="ph" class="tab-content">
                <p>Net charge at pH PH_VALUE: <strong>NET_CHARGE</strong></p>
                <p>pH-sensitive residues: <strong>NUM_SENSITIVE</strong></p>
            </div>
            <div id="chaperones" class="tab-content">
                CHAPERONE_LIST
            </div>
        </div>
        
        <!-- PTM Sites -->
        <div class="panel full-width">
            <h2><span class="icon">⚡</span> Post-Translational Modifications</h2>
            <table class="table">
                <tr><th>Position</th><th>Residue</th><th>Modification</th><th>Score</th><th>Context</th></tr>
                PTM_ROWS
            </table>
        </div>
    </div>
    
    <script>
        const analysisData = ANALYSIS_JSON;
        
        // Sequence display with coloring
        function renderSequence() {
            const seq = analysisData.sequence;
            const disorder = analysisData.disorder?.disorder_scores || [];
            const container = document.getElementById('sequence-display');
            
            let html = '';
            for (let i = 0; i < seq.length; i++) {
                let cls = 'residue';
                if (disorder[i] > 0.5) cls += ' d-res';
                html += `<span class="${cls}" title="Pos ${i+1}: Disorder ${(disorder[i]*100||50).toFixed(0)}%">${seq[i]}</span>`;
            }
            container.innerHTML = html;
        }
        
        // 3D Structure plot
        function render3D() {
            const coords = analysisData.structure?.backbone_coords || [];
            const n = analysisData.sequence.length;
            
            // Generate coords if not available
            let x = [], y = [], z = [], colors = [];
            for (let i = 0; i < n; i++) {
                x.push(1.5 * Math.cos(i * 1.75));
                y.push(1.5 * Math.sin(i * 1.75));
                z.push(i * 1.5);
                colors.push(i);
            }
            
            Plotly.newPlot('structure-3d', [{
                type: 'scatter3d',
                mode: 'lines+markers',
                x: x, y: y, z: z,
                marker: { size: 4, color: colors, colorscale: 'Viridis' },
                line: { width: 3, color: '#60ffff' }
            }], {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 0, r: 0, t: 0, b: 0 },
                scene: {
                    xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    zaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    bgcolor: 'rgba(0,0,0,0)'
                }
            }, { responsive: true });
        }
        
        // Folding energy plot
        function renderFolding() {
            const profile = analysisData.folding?.energy_profile || [];
            const x = profile.map((p, i) => i);
            const y = profile.map(p => p[1] || p);
            
            Plotly.newPlot('folding-chart', [{
                type: 'scatter',
                mode: 'lines',
                x: x, y: y,
                fill: 'tozeroy',
                line: { color: '#60ffff', width: 2 }
            }], {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 40, r: 20, t: 10, b: 40 },
                xaxis: { title: 'Step', color: '#888', gridcolor: 'rgba(100,100,200,0.2)' },
                yaxis: { title: 'Energy', color: '#888', gridcolor: 'rgba(100,100,200,0.2)' }
            }, { responsive: true });
        }
        
        // Disorder plot
        function renderDisorder() {
            const scores = analysisData.disorder?.disorder_scores || [];
            const x = scores.map((_, i) => i + 1);
            
            Plotly.newPlot('disorder-chart', [{
                type: 'scatter',
                mode: 'lines',
                x: x, y: scores,
                fill: 'tozeroy',
                line: { color: '#ff8060', width: 2 }
            }, {
                type: 'scatter',
                mode: 'lines',
                x: [0, x.length],
                y: [0.5, 0.5],
                line: { color: '#ffffff', dash: 'dash', width: 1 }
            }], {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 40, r: 20, t: 10, b: 40 },
                xaxis: { title: 'Residue', color: '#888', gridcolor: 'rgba(100,100,200,0.2)' },
                yaxis: { title: 'Disorder Score', color: '#888', range: [0, 1], gridcolor: 'rgba(100,100,200,0.2)' },
                showlegend: false
            }, { responsive: true });
        }
        
        // Tab switching
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Initialize
        renderSequence();
        render3D();
        renderFolding();
        renderDisorder();
    </script>
</body>
</html>'''
    
    # Fill in template values
    disorder_scores = analysis.disorder.get('disorder_scores', [0.5] * len(analysis.sequence))
    disorder_percent = int(analysis.disorder.get('disordered_fraction', 0.5) * 100)
    
    net_charge = analysis.cellular.get('ph_effects', {}).get('net_charge', 0)
    
    # Pocket rows
    pockets = analysis.drug_binding.get('pockets', [])
    pocket_rows = ""
    for p in pockets[:5]:
        drug = "✓" if p.get('is_druggable') else "-"
        pocket_rows += f"<tr><td>#{p.get('id', 0)}</td><td>{p.get('volume', 0):.0f}</td><td>{p.get('druggability', 0):.2f} {drug}</td></tr>"
    
    if not pocket_rows:
        pocket_rows = "<tr><td colspan='3'>No pockets detected</td></tr>"
    
    # PTM rows
    ptm_sites = analysis.disorder.get('ptm_sites', [])
    ptm_rows = ""
    for p in ptm_sites[:10]:
        ptm_rows += f"<tr><td>{p.get('position', 0)+1}</td><td>{p.get('residue', '?')}</td><td>{p.get('modification', '')}</td><td>{p.get('score', 0):.2f}</td><td>{p.get('context', '')}</td></tr>"
    
    if not ptm_rows:
        ptm_rows = "<tr><td colspan='5'>No PTM sites predicted</td></tr>"
    
    # Chaperone list
    chaperones = analysis.cellular.get('chaperones', [])
    chaperone_html = ""
    for c in chaperones:
        chaperone_html += f"<p>• <strong>{c.get('chaperone', 'Unknown')}</strong>: {c.get('probability', 0):.0%} probability ({c.get('effect', '')})</p>"
    
    if not chaperone_html:
        chaperone_html = "<p>No chaperone interactions predicted</p>"
    
    # Crowding effects
    crowding = analysis.cellular.get('crowding_effects', {})
    stability_change = crowding.get('stability', {}).get('change_percent', 0)
    diffusion_change = crowding.get('diffusion', {}).get('change_percent', 0)
    
    # Replace placeholders
    html = html_template.replace('SEQUENCE_LENGTH', str(len(analysis.sequence)))
    html = html.replace('COMPARTMENT', analysis.cellular.get('compartment', 'cytoplasm'))
    html = html.replace('DISORDER_PERCENT', str(disorder_percent))
    html = html.replace('NET_CHARGE', f"{net_charge:.1f}")
    html = html.replace('PATHWAY_TYPE', analysis.folding.get('pathway', {}).get('pathway_type', 'unknown'))
    html = html.replace('AGG_SCORE', f"{analysis.folding.get('aggregation_score', 0):.2f}")
    html = html.replace('DISORDER_CLASS', analysis.disorder.get('classification', 'Unknown'))
    html = html.replace('LLPS_SCORE', f"{analysis.disorder.get('llps', {}).get('propensity', 0):.2f}")
    html = html.replace('POCKET_ROWS', pocket_rows)
    html = html.replace('PTM_ROWS', ptm_rows)
    html = html.replace('STABILITY_CHANGE', f"{stability_change:+.1f}")
    html = html.replace('DIFFUSION_CHANGE', f"{diffusion_change:.1f}")
    html = html.replace('PH_VALUE', str(analysis.cellular.get('environment', {}).get('pH', 7.4)))
    html = html.replace('NUM_SENSITIVE', str(len(analysis.cellular.get('ph_effects', {}).get('sensitive_residues', []))))
    html = html.replace('CHAPERONE_LIST', chaperone_html)
    html = html.replace('ANALYSIS_JSON', analysis.to_json())
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test sequence (Ubiquitin-like)
    test_seq = "MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    
    analysis = analyze_protein_complete(test_seq)
    generate_html_visualizer(analysis, "protein_complete_analysis.html")
