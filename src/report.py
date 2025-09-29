# src/report.py
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class NpEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy data types in JSON"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, dict):
            # Convert numpy int64 keys to regular int keys
            return {str(k) if isinstance(k, (np.integer, np.int64)) else k: v for k, v in obj.items()}
        return super().default(obj)


def generate_report(out_dir, images, tables, summary, metrics_paths, interactive_plots=None):
    """
    Generate a professional HTML report with embedded interactive plots.
    """
    out_dir = Path(out_dir)
    report_path = out_dir / "report.html"

    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head>")
    html.append("<meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("<title>eDNA Analysis Report</title>")
    html.append("<link href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css' rel='stylesheet'>")
    html.append("<style>")
    
    # Enhanced CSS with scientific styling
    html.append("""
        :root {
            --primary-color: #10b981;
            --secondary-color: #3b82f6;
            --accent-color: #06b6d4;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --background: #ffffff;
            --surface: #f8fafc;
            --border: #e5e7eb;
            --border-light: #f3f4f6;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header Styles */
        .header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            opacity: 0.3;
        }

        .header-content {
            position: relative;
            z-index: 1;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 1rem;
        }

        .report-meta {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            font-size: 0.9rem;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Section Styles */
        .section {
            background: var(--background);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-light);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border-light);
        }

        .section-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--secondary-color), var(--accent-color));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
        }

        h2 {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--primary-color);
        }

        h3 {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 1.5rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(135deg, var(--secondary-color), var(--accent-color));
            border-radius: 2px;
        }

        /* Table Styles */
        .table-container {
            background: var(--background);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
            margin: 1.5rem 0;
            border: 1px solid var(--border);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }

        th {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            font-weight: 600;
            padding: 1rem;
            text-align: left;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        td {
            padding: 0.875rem 1rem;
            border-bottom: 1px solid var(--border-light);
            transition: background-color 0.2s ease;
        }

        tr:hover {
            background-color: var(--surface);
        }

        tr:last-child td {
            border-bottom: none;
        }

        /* Alternating row colors */
        tbody tr:nth-child(even) {
            background-color: #fafbfc;
        }

        /* Number formatting */
        .number {
            font-family: 'Monaco', 'Menlo', monospace;
            font-weight: 500;
        }

        /* Summary table special styling */
        .summary-table th {
            background: linear-gradient(135deg, var(--success-color), #059669);
            min-width: 200px;
        }

        .summary-table td {
            font-weight: 500;
        }

        /* Image Styles */
        .image-container {
            background: var(--background);
            border-radius: 8px;
            padding: 1rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            text-align: center;
        }

        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .image-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        /* Interactive plots */
        .plot-container {
            background: var(--background);
            border-radius: 8px;
            padding: 1rem;
            margin: 1.5rem 0;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }

        /* Metrics cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }

        .metric-card {
            background: var(--background);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-color);
        }

        .metric-card h4 {
            color: var(--primary-color);
            font-size: 1.2rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        /* Status indicators */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-success {
            background: #dcfce7;
            color: #166534;
        }

        .status-warning {
            background: #fef3c7;
            color: #92400e;
        }

        .status-info {
            background: #dbeafe;
            color: #1e40af;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .header {
                padding: 2rem 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .section {
                padding: 1.5rem;
            }
            
            .report-meta {
                gap: 1rem;
            }
            
            .table-container {
                overflow-x: auto;
            }
        }

        /* Loading animation for interactive elements */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            color: var(--text-secondary);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }
    """)
    
    html.append("</style></head><body>")
    html.append("<div class='container'>")
    
    # Enhanced header
    html.append("<div class='header'>")
    html.append("<div class='header-content'>")
    html.append("<h1><i class='fas fa-dna'></i> eDNA Analysis Report</h1>")
    html.append("<div class='subtitle'>Comprehensive Environmental DNA Analysis Results</div>")
    html.append("<div class='report-meta'>")
    html.append(f"<div class='meta-item'><i class='fas fa-calendar'></i> Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</div>")
    html.append("<div class='meta-item'><i class='fas fa-microscope'></i> Analysis Type: eDNA Sequencing</div>")
    html.append("</div>")
    html.append("</div>")
    html.append("</div>")

    # Summary section with enhanced styling
    html.append("<div class='section'>")
    html.append("<div class='section-header'>")
    html.append("<div class='section-icon'><i class='fas fa-chart-pie'></i></div>")
    html.append("<h2>Executive Summary</h2>")
    html.append("</div>")
    html.append("<div class='table-container'>")
    html.append("<table class='summary-table'>")
    for k, v in summary.items():
        # Format values nicely
        if isinstance(v, (int, float)):
            if isinstance(v, float):
                formatted_v = f"<span class='number'>{v:,.4f}</span>" if v < 1 else f"<span class='number'>{v:,.2f}</span>"
            else:
                formatted_v = f"<span class='number'>{v:,}</span>"
        else:
            formatted_v = str(v)
        html.append(f"<tr><th>{k.replace('_', ' ').title()}</th><td>{formatted_v}</td></tr>")
    html.append("</table>")
    html.append("</div>")
    html.append("</div>")

    # Interactive Plots Section
    html.append("<div class='section'>")
    html.append("<div class='section-header'>")
    html.append("<div class='section-icon'><i class='fas fa-chart-line'></i></div>")
    html.append("<h2>Interactive Visualizations</h2>")
    html.append("</div>")
    
    if interactive_plots:
        html.append("<h3><i class='fas fa-chart-pie'></i> Final Status Distribution</h3>")
        html.append("<div class='plot-container'>")
        html.append(interactive_plots.get('status_pie', '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Pie chart not available.</div>'))
        html.append("</div>")
        
        html.append("<h3><i class='fas fa-project-diagram'></i> Interactive UMAP Projection</h3>")
        html.append("<div class='plot-container'>")
        html.append(interactive_plots.get('umap', '<div class="loading"><i class="fas fa-spinner fa-spin"></i> UMAP visualization not available.</div>'))
        html.append("</div>")
        
        html.append("<h3><i class='fas fa-chart-bar'></i> Species Abundance Analysis</h3>")
        html.append("<div class='plot-container'>")
        html.append(interactive_plots.get('abundance_bar', '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Abundance chart not available.</div>'))
        html.append("</div>")
    else:
        html.append("<div class='status-badge status-warning'>")
        html.append("<i class='fas fa-exclamation-triangle'></i> No interactive plots were generated")
        html.append("</div>")
    html.append("</div>")
    
    # Static Plots section
    html.append("<div class='section'>")
    html.append("<div class='section-header'>")
    html.append("<div class='section-icon'><i class='fas fa-images'></i></div>")
    html.append("<h2>Static Plots & Figures</h2>")
    html.append("</div>")
    
    if images:
        for img_path in images:
            if Path(img_path).exists():
                html.append("<div class='image-container'>")
                html.append(f"<div class='image-title'><i class='fas fa-image'></i> {Path(img_path).stem.replace('_', ' ').title()}</div>")
                html.append(f"<img src='{Path(img_path).name}' alt='{Path(img_path).name}' loading='lazy'>")
                html.append("</div>")
            else:
                html.append(f"<div class='status-badge status-warning'><i class='fas fa-exclamation-triangle'></i> {Path(img_path).name} not found</div>")
    else:
        html.append("<div class='status-badge status-info'><i class='fas fa-info-circle'></i> No static plots available</div>")
    html.append("</div>")

    # Metrics section with cards
    html.append("<div class='section'>")
    html.append("<div class='section-header'>")
    html.append("<div class='section-icon'><i class='fas fa-calculator'></i></div>")
    html.append("<h2>Performance Metrics</h2>")
    html.append("</div>")
    
    if metrics_paths:
        html.append("<div class='metrics-grid'>")
        for metric_name, metric_path in metrics_paths.items():
            html.append("<div class='metric-card'>")
            html.append(f"<h4><i class='fas fa-chart-area'></i> {metric_name.replace('_', ' ').title()}</h4>")
            
            if Path(metric_path).exists():
                with open(metric_path) as f:
                    metrics = json.load(f)
                
                html.append("<div class='table-container'>")
                html.append("<table>")
                
                if metrics and isinstance(list(metrics.values())[0], dict):
                    # Multi-level metrics (like per-rank metrics)
                    headers = list(list(metrics.values())[0].keys())
                    html.append("<tr><th>Rank/Level</th>" + "".join(f"<th>{h.replace('_', ' ').title()}</th>" for h in headers) + "</tr>")
                    for rank, rank_metrics in metrics.items():
                        html.append(f"<tr><th>{rank}</th>")
                        for header in headers:
                            value = rank_metrics.get(header, 0)
                            if isinstance(value, (int, float)):
                                formatted_value = f"<span class='number'>{value:.4f}</span>" if isinstance(value, float) else f"<span class='number'>{value:,}</span>"
                            else:
                                formatted_value = str(value)
                            html.append(f"<td>{formatted_value}</td>")
                        html.append("</tr>")
                else:
                    # Simple key-value metrics
                    html.append("<tr><th>Metric</th><th>Value</th></tr>")
                    for metric_key, metric_val in metrics.items():
                        formatted_key = metric_key.replace('_', ' ').title()
                        if isinstance(metric_val, (float, int)):
                            if isinstance(metric_val, float):
                                formatted_val = f"<span class='number'>{metric_val:.4f}</span>"
                            else:
                                formatted_val = f"<span class='number'>{metric_val:,}</span>"
                        else:
                            formatted_val = str(metric_val)
                        html.append(f"<tr><td>{formatted_key}</td><td>{formatted_val}</td></tr>")
                
                html.append("</table>")
                html.append("</div>")
            else:
                html.append(f"<div class='status-badge status-warning'><i class='fas fa-exclamation-triangle'></i> Metrics file not found</div>")
            
            html.append("</div>")
        html.append("</div>")
    else:
        html.append("<div class='status-badge status-info'><i class='fas fa-info-circle'></i> No metrics available</div>")
    html.append("</div>")

    # Tables section
    html.append("<div class='section'>")
    html.append("<div class='section-header'>")
    html.append("<div class='section-icon'><i class='fas fa-table'></i></div>")
    html.append("<h2>Data Tables</h2>")
    html.append("</div>")
    
    if tables:
        for t in tables:
            if Path(t).exists():
                df = pd.read_csv(t)
                html.append(f"<h3><i class='fas fa-file-csv'></i> {Path(t).stem.replace('_', ' ').title()}</h3>")
                html.append(f"<div class='status-badge status-info'><i class='fas fa-info-circle'></i> {len(df)} rows × {len(df.columns)} columns</div>")
                html.append("<div class='table-container'>")
                
                # Convert DataFrame to HTML with better formatting
                table_html = df.to_html(
                    index=False, 
                    escape=False, 
                    table_id=None, 
                    classes=None, 
                    border=0,
                    float_format=lambda x: f'{x:.4f}' if pd.notnull(x) and isinstance(x, (int, float)) else x
                )
                html.append(table_html)
                html.append("</div>")
            else:
                html.append(f"<div class='status-badge status-warning'><i class='fas fa-exclamation-triangle'></i> Table {Path(t).name} not found</div>")
    else:
        html.append("<div class='status-badge status-info'><i class='fas fa-info-circle'></i> No data tables available</div>")
    html.append("</div>")

    # Footer
    html.append("<div class='footer'>")
    html.append("<p><i class='fas fa-dna'></i> Generated by eDNA Analysis Pipeline | ")
    html.append(f"<i class='fas fa-clock'></i> Report created on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>")
    html.append("</div>")
    
    html.append("</div>") # Close container
    html.append("</body></html>")

    # Save HTML
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))

    return str(report_path)