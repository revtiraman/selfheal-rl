"""Service mesh HTML/CSS visualization for Gradio UI."""

from __future__ import annotations

from typing import Dict, List, Optional

from config import SERVICES, SERVICE_NAMES

# Layout: row assignments for the dependency tree
SERVICE_LAYOUT = {
    "api-gateway":          {"row": 0, "col": 1},
    "auth-service":         {"row": 1, "col": 0},
    "search-service":       {"row": 1, "col": 2},
    "payment-service":      {"row": 2, "col": 0},
    "order-service":        {"row": 2, "col": 1},
    "notification-service": {"row": 2, "col": 2},
    "user-db":              {"row": 3, "col": 0},
    "cache-layer":          {"row": 3, "col": 1},
    "restaurant-db":        {"row": 3, "col": 2},
    "order-db":             {"row": 3, "col": 3},
}

CSS = """
<style>
.mesh-container {
    background: #0f0f23;
    border-radius: 12px;
    padding: 24px;
    font-family: 'Courier New', monospace;
    min-height: 500px;
    position: relative;
}
.mesh-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    max-width: 900px;
    margin: 0 auto;
}
.service-box {
    background: #1a1a3e;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
}
.service-box.healthy {
    border-color: #00ff88;
    box-shadow: 0 0 12px rgba(0,255,136,0.3);
}
.service-box.degraded {
    border-color: #ffaa00;
    box-shadow: 0 0 12px rgba(255,170,0,0.4);
    animation: pulse-yellow 1.5s infinite;
}
.service-box.down {
    border-color: #ff3366;
    box-shadow: 0 0 16px rgba(255,51,102,0.5);
    animation: shake 0.5s infinite;
}
.service-box.recovering {
    border-color: #44aaff;
    box-shadow: 0 0 12px rgba(68,170,255,0.4);
    animation: pulse-blue 1.2s infinite;
}
.service-box.targeted {
    outline: 3px solid #fff;
    outline-offset: 3px;
}
.svc-name {
    font-size: 11px;
    font-weight: bold;
    color: #e0e0ff;
    margin-bottom: 4px;
}
.svc-status {
    font-size: 18px;
    margin: 4px 0;
}
.svc-metrics {
    font-size: 10px;
    color: #888;
}
.metric-bar {
    background: #333;
    border-radius: 3px;
    height: 4px;
    margin: 2px 0;
    overflow: hidden;
}
.metric-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}
.metric-fill.cpu { background: #ff6644; }
.metric-fill.mem { background: #44aaff; }
.action-label {
    position: absolute;
    top: -10px;
    right: -10px;
    background: #ffaa00;
    color: #000;
    font-size: 10px;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 4px;
    animation: fadeIn 0.3s;
}
.action-label.success { background: #00ff88; }
.action-label.fail { background: #ff3366; color: #fff; }
@keyframes pulse-yellow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
@keyframes pulse-blue {
    0%, 100% { box-shadow: 0 0 12px rgba(68,170,255,0.4); }
    50% { box-shadow: 0 0 24px rgba(68,170,255,0.7); }
}
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-3px); }
    75% { transform: translateX(3px); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1); }
}
.mesh-stats {
    display: flex;
    justify-content: space-around;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #333;
}
.stat-item {
    text-align: center;
    color: #aaa;
    font-size: 12px;
}
.stat-value {
    font-size: 20px;
    font-weight: bold;
    color: #fff;
}
.row-label {
    color: #555;
    font-size: 10px;
    text-align: center;
    padding: 4px 0;
    grid-column: 1 / -1;
}
</style>
"""


def render_mesh(
    service_statuses: Dict[str, dict],
    action_info: Optional[dict] = None,
    step: int = 0,
    total_reward: float = 0.0,
    actions_remaining: int = 10,
) -> str:
    """Generate HTML for the service mesh visualization."""
    html = CSS + '<div class="mesh-container">'

    # Build grid — 4 rows
    html += '<div class="mesh-grid">'

    rows = {0: [], 1: [], 2: [], 3: []}
    for name, layout in SERVICE_LAYOUT.items():
        rows[layout["row"]].append((name, layout["col"]))

    row_labels = ["Gateway Layer", "Service Layer", "Application Layer", "Data Layer"]

    for row_idx in range(4):
        html += f'<div class="row-label">{row_labels[row_idx]}</div>'
        items = sorted(rows[row_idx], key=lambda x: x[1])

        # Pad grid to 4 columns
        col_map = {item[1]: item[0] for item in items}
        for col in range(4):
            if col in col_map:
                name = col_map[col]
                html += _render_service_box(name, service_statuses.get(name, {}), action_info)
            else:
                html += '<div></div>'

    html += '</div>'

    # Stats bar
    total = len(SERVICE_NAMES)
    healthy = sum(1 for s in service_statuses.values() if s.get("status", 0) >= 0.9)
    degraded = sum(1 for s in service_statuses.values() if 0.1 < s.get("status", 0) < 0.9)
    down = sum(1 for s in service_statuses.values() if s.get("status", 0) <= 0.1)
    health_pct = healthy / total * 100

    html += f'''
    <div class="mesh-stats">
        <div class="stat-item">
            <div class="stat-value" style="color:#00ff88">{healthy}</div>
            Healthy
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color:#ffaa00">{degraded}</div>
            Degraded
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color:#ff3366">{down}</div>
            Down
        </div>
        <div class="stat-item">
            <div class="stat-value">{health_pct:.0f}%</div>
            System Health
        </div>
        <div class="stat-item">
            <div class="stat-value">Step {step}</div>
            Progress
        </div>
        <div class="stat-item">
            <div class="stat-value">{total_reward:+.0f}</div>
            Reward
        </div>
        <div class="stat-item">
            <div class="stat-value">{actions_remaining}</div>
            Actions Left
        </div>
    </div>
    '''

    html += '</div>'
    return html


def _render_service_box(name: str, status_data: dict, action_info: Optional[dict]) -> str:
    status_val = status_data.get("status", 1.0)
    recovering = status_data.get("recovering", False)
    cpu = status_data.get("cpu", 0)
    memory = status_data.get("memory", 0)
    latency = status_data.get("latency", 0)
    error_rate = status_data.get("error_rate", 0)

    if recovering:
        css_class = "recovering"
        icon = "🔄"
    elif status_val >= 0.9:
        css_class = "healthy"
        icon = "🟢"
    elif status_val > 0.1:
        css_class = "degraded"
        icon = "🟡"
    else:
        css_class = "down"
        icon = "🔴"

    targeted = ""
    action_label = ""
    if action_info and action_info.get("target") == name:
        targeted = " targeted"
        act = action_info.get("action_type", "")
        success = action_info.get("success", False)
        label_class = "success" if success else "fail"
        label_icon = "✅" if success else "❌"
        action_label = f'<div class="action-label {label_class}">{act} {label_icon}</div>'

    return f'''
    <div class="service-box {css_class}{targeted}">
        {action_label}
        <div class="svc-name">{name}</div>
        <div class="svc-status">{icon}</div>
        <div class="svc-metrics">
            CPU <div class="metric-bar"><div class="metric-fill cpu" style="width:{cpu*100:.0f}%"></div></div>
            MEM <div class="metric-bar"><div class="metric-fill mem" style="width:{memory*100:.0f}%"></div></div>
            {latency:.0f}ms | err:{error_rate:.0%}
        </div>
    </div>
    '''


def render_action_log(history: list) -> str:
    """Render action log as HTML."""
    if not history:
        return '<div style="color:#666;text-align:center;padding:20px;">No actions yet</div>'

    html = '<div style="font-family:monospace;font-size:12px;max-height:300px;overflow-y:auto;background:#0f0f23;padding:12px;border-radius:8px;">'
    for record in history:
        step = record.step
        action = f"{record.action_type}({record.target_service})"
        if record.action_success:
            color = "#00ff88"
            icon = "✅"
        else:
            color = "#ff3366"
            icon = "❌"
        reward_color = "#00ff88" if record.reward >= 0 else "#ff3366"
        html += f'<div style="padding:3px 0;border-bottom:1px solid #222;">'
        html += f'<span style="color:#666;">Step {step:2d}</span> '
        html += f'<span style="color:{color};">{icon} {action}</span> '
        html += f'<span style="color:{reward_color};float:right;">reward: {record.reward:+.1f}</span>'
        html += f'</div>'
    html += '</div>'
    return html
