"""SelfHealRL — Gradio Web Interface."""

from __future__ import annotations

import os
import sys
import time

import gradio as gr
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ACTION_TYPES, MAX_STEPS_PER_EPISODE, SERVICE_NAMES
from core.graders import Grader
from core.llm_scorer import LLMScorer
from core.metrics import EpisodeMetrics
from env.selfheal_env import SelfHealEnv
from ui.visualizer import render_action_log, render_mesh
from ui.replay import generate_replay_frames, generate_comparison_frames

# ── Global state ──────────────────────────────────────────────

_env: SelfHealEnv | None = None
_model = None
_scorer = LLMScorer(mode="heuristic")


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from stable_baselines3 import PPO
        for path in ["models/selfheal_agent_final.zip", "models/phase1_easy.zip"]:
            if os.path.exists(path):
                _model = PPO.load(path)
                return _model
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# TAB 1: Live Demo
# ═══════════════════════════════════════════════════════════════

def init_scenario(difficulty):
    global _env
    _env = SelfHealEnv(difficulty=difficulty, partial_observability=True)
    obs, info = _env.reset()
    statuses = _env.mesh.get_all_statuses()
    html = render_mesh(statuses, step=0)
    return html, "Scenario generated! System has failures. Run an agent to fix them.", ""


def run_agent_demo(difficulty, agent_type):
    global _env
    _env = SelfHealEnv(difficulty=difficulty, partial_observability=(agent_type == "Trained Agent"))
    obs, info = _env.reset()

    model = _load_model() if agent_type == "Trained Agent" else None

    frames = []
    done = False
    for step in range(MAX_STEPS_PER_EPISODE):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = _env.action_space.sample()

        obs, reward, term, trunc, info = _env.step(action)
        done = term or trunc

        frames.append(_env.episode_history[-1])
        if done:
            break

    # Build final visualization
    history = _env.episode_history
    statuses = _env.mesh.get_all_statuses()
    last = history[-1] if history else None

    action_info = None
    if last:
        action_info = {
            "target": last.target_service,
            "action_type": last.action_type,
            "success": last.action_success,
        }

    mesh_html = render_mesh(
        statuses,
        action_info=action_info,
        step=len(history),
        total_reward=_env.total_reward,
        actions_remaining=_env.actions_remaining,
    )
    log_html = render_action_log(history)

    summary = _env.get_episode_summary()
    grades = Grader.grade_all(summary)

    status_msg = f"{'✅ RECOVERED' if _env.mesh.is_fully_recovered() else '❌ NOT RECOVERED'} "
    status_msg += f"in {len(history)} steps | Reward: {_env.total_reward:.1f} | "
    status_msg += f"Grade: {grades['overall_score']:.2f}"

    return mesh_html, status_msg, log_html


def run_step_by_step(difficulty):
    """Generator that yields frame-by-frame for animated replay."""
    env = SelfHealEnv(difficulty=difficulty, partial_observability=True)
    obs, info = env.reset()
    model = _load_model()

    all_records = []
    for step in range(MAX_STEPS_PER_EPISODE):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, term, trunc, info = env.step(action)
        record = env.episode_history[-1]
        all_records.append(record)

        mesh_html = render_mesh(
            record.system_state,
            action_info={"target": record.target_service, "action_type": record.action_type, "success": record.action_success},
            step=step,
            total_reward=env.total_reward,
            actions_remaining=env.actions_remaining,
        )
        log_html = render_action_log(all_records)
        status = f"Step {step} | Health: {record.system_health:.0%} | Reward: {env.total_reward:.1f}"

        yield mesh_html, status, log_html

        if term or trunc:
            break


# ═══════════════════════════════════════════════════════════════
# TAB 2: Agent vs Random
# ═══════════════════════════════════════════════════════════════

def run_comparison(difficulty):
    seed = np.random.randint(0, 10000)

    # Trained agent
    env_t = SelfHealEnv(difficulty=difficulty, partial_observability=True)
    obs_t, _ = env_t.reset(seed=seed)
    model = _load_model()

    done = False
    for _ in range(MAX_STEPS_PER_EPISODE):
        if model:
            action, _ = model.predict(obs_t, deterministic=True)
            action = int(action)
        else:
            action = env_t.action_space.sample()
        obs_t, _, term, trunc, _ = env_t.step(action)
        if term or trunc:
            break

    # Random agent (same seed = same scenario)
    env_r = SelfHealEnv(difficulty=difficulty, partial_observability=False)
    obs_r, _ = env_r.reset(seed=seed)

    done = False
    for _ in range(MAX_STEPS_PER_EPISODE):
        action = env_r.action_space.sample()
        obs_r, _, term, trunc, _ = env_r.step(action)
        if term or trunc:
            break

    # Results
    t_statuses = env_t.mesh.get_all_statuses()
    r_statuses = env_r.mesh.get_all_statuses()

    t_html = render_mesh(t_statuses, step=len(env_t.episode_history), total_reward=env_t.total_reward)
    r_html = render_mesh(r_statuses, step=len(env_r.episode_history), total_reward=env_r.total_reward)

    t_summary = env_t.get_episode_summary()
    r_summary = env_r.get_episode_summary()
    t_grades = Grader.grade_all(t_summary)
    r_grades = Grader.grade_all(r_summary)

    comparison = f"""
| Metric | {'Trained Agent' if model else 'Agent A'} | Random Agent |
|--------|:------:|:------:|
| Steps | {len(env_t.episode_history)} | {len(env_r.episode_history)} |
| Reward | {env_t.total_reward:.1f} | {env_r.total_reward:.1f} |
| Recovered | {'✅' if env_t.mesh.is_fully_recovered() else '❌'} | {'✅' if env_r.mesh.is_fully_recovered() else '❌'} |
| Health | {env_t.mesh.system_health():.0%} | {env_r.mesh.system_health():.0%} |
| Grade | {t_grades['overall_score']:.2f} | {r_grades['overall_score']:.2f} |
"""

    return t_html, r_html, comparison


# ═══════════════════════════════════════════════════════════════
# TAB 3: Grading Report
# ═══════════════════════════════════════════════════════════════

def run_grading(difficulty, num_episodes):
    num_episodes = int(num_episodes)
    model = _load_model()

    all_grades = []
    for i in range(num_episodes):
        env = SelfHealEnv(difficulty=difficulty, partial_observability=True)
        obs, _ = env.reset(seed=i)
        done = False
        for _ in range(MAX_STEPS_PER_EPISODE):
            if model:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        summary = env.get_episode_summary()
        grades = Grader.grade_all(summary)
        all_grades.append(grades)

    # Aggregate
    grader_names = ["recovery", "mttr", "cascade_prevention", "dependency_ordering", "efficiency", "diagnosis"]
    report = "## Grading Report\n\n"
    report += f"**{num_episodes} episodes on {difficulty} difficulty**\n"
    report += f"**Agent: {'Trained' if model else 'Random (no trained model found)'}**\n\n"
    report += "| Grader | Avg Score | Pass Rate |\n|--------|:---------:|:---------:|\n"

    for name in grader_names:
        scores = [g[name]["score"] for g in all_grades]
        passes = [g[name]["passed"] for g in all_grades]
        avg = np.mean(scores)
        pass_rate = np.mean(passes)
        icon = "✅" if pass_rate >= 0.8 else "⚠️" if pass_rate >= 0.5 else "❌"
        report += f"| {icon} {name} | {avg:.2f} | {pass_rate:.0%} |\n"

    overall_scores = [g["overall_score"] for g in all_grades]
    overall_pass = [g["overall_pass"] for g in all_grades]
    report += f"\n**Overall Score: {np.mean(overall_scores):.2f}** | "
    report += f"**Overall Pass Rate: {np.mean(overall_pass):.0%}**"

    return report


# ═══════════════════════════════════════════════════════════════
# TAB 4: LLM Analysis
# ═══════════════════════════════════════════════════════════════

def run_llm_analysis(difficulty):
    model = _load_model()
    env = SelfHealEnv(difficulty=difficulty, partial_observability=True)
    obs, _ = env.reset(seed=42)

    done = False
    for _ in range(MAX_STEPS_PER_EPISODE):
        if model:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    summary = env.get_episode_summary()
    episode_scores = _scorer.score_episode(summary)
    strategy = _scorer.score_strategy(summary)

    report = "## LLM Decision Analysis\n\n"
    report += f"**Scenario:** {summary['scenario']}\n\n"

    report += "### Critical Decisions\n\n"
    for dec in episode_scores["critical_decisions"]:
        report += f"**Step {dec['step']}: {dec['action']}**\n"
        report += f"- Root cause: {dec['root_cause_score']}/10\n"
        report += f"- Dependencies: {dec['dependency_score']}/10\n"
        report += f"- Action type: {dec['action_type_score']}/10\n"
        report += f"- Timing: {dec['timing_score']}/10\n"
        report += f"- Overall: **{dec['overall_score']}/10**\n"
        if "reasoning" in dec:
            report += f"- _{dec['reasoning']}_\n"
        report += "\n"

    report += "### Overall Strategy\n\n"
    report += f"| Criterion | Score |\n|-----------|:-----:|\n"
    report += f"| Diagnostic Approach | {strategy['diagnostic_approach']}/10 |\n"
    report += f"| Prioritization | {strategy['prioritization']}/10 |\n"
    report += f"| Patience | {strategy['patience']}/10 |\n"
    report += f"| **Overall Strategy** | **{strategy['overall_strategy']}/10** |\n"

    return report


# ═══════════════════════════════════════════════════════════════
# Build Gradio App
# ═══════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="SelfHealRL — Autonomous Microservices Recovery",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=".gradio-container { max-width: 1200px !important; }",
    ) as demo:
        gr.Markdown("# 🏥 SelfHealRL — Autonomous Microservices Recovery\n> An RL agent that learns to diagnose and fix cascading failures in microservices")

        with gr.Tabs():
            # ── Tab 1: Live Demo ──
            with gr.Tab("🔴 Live Demo"):
                with gr.Row():
                    diff_select = gr.Dropdown(
                        choices=["EASY", "MEDIUM", "HARD", "CHAOS"],
                        value="MEDIUM",
                        label="Difficulty",
                    )
                    agent_select = gr.Dropdown(
                        choices=["Trained Agent", "Random Agent"],
                        value="Trained Agent",
                        label="Agent Type",
                    )
                    run_btn = gr.Button("▶️ Run Demo", variant="primary")

                mesh_display = gr.HTML(label="Service Mesh")
                status_text = gr.Markdown("Click 'Run Demo' to start")
                action_log = gr.HTML(label="Action Log")

                run_btn.click(
                    fn=run_agent_demo,
                    inputs=[diff_select, agent_select],
                    outputs=[mesh_display, status_text, action_log],
                )

            # ── Tab 2: Agent vs Random ──
            with gr.Tab("⚔️ Agent vs Random"):
                with gr.Row():
                    cmp_diff = gr.Dropdown(
                        choices=["EASY", "MEDIUM", "HARD", "CHAOS"],
                        value="MEDIUM",
                        label="Difficulty",
                    )
                    cmp_btn = gr.Button("⚔️ Run Comparison", variant="primary")

                with gr.Row():
                    trained_display = gr.HTML(label="Trained Agent")
                    random_display = gr.HTML(label="Random Agent")

                cmp_results = gr.Markdown()

                cmp_btn.click(
                    fn=run_comparison,
                    inputs=[cmp_diff],
                    outputs=[trained_display, random_display, cmp_results],
                )

            # ── Tab 3: Grading Report ──
            with gr.Tab("📋 Grading Report"):
                with gr.Row():
                    grade_diff = gr.Dropdown(
                        choices=["EASY", "MEDIUM", "HARD", "CHAOS"],
                        value="MEDIUM",
                        label="Difficulty",
                    )
                    grade_n = gr.Slider(
                        minimum=5, maximum=50, value=10, step=5,
                        label="Number of Episodes",
                    )
                    grade_btn = gr.Button("📊 Run Evaluation", variant="primary")

                grade_output = gr.Markdown()
                grade_btn.click(
                    fn=run_grading,
                    inputs=[grade_diff, grade_n],
                    outputs=[grade_output],
                )

            # ── Tab 4: LLM Analysis ──
            with gr.Tab("🧠 LLM Analysis"):
                with gr.Row():
                    llm_diff = gr.Dropdown(
                        choices=["EASY", "MEDIUM", "HARD", "CHAOS"],
                        value="MEDIUM",
                        label="Difficulty",
                    )
                    llm_btn = gr.Button("🧠 Analyze Episode", variant="primary")

                llm_output = gr.Markdown()
                llm_btn.click(
                    fn=run_llm_analysis,
                    inputs=[llm_diff],
                    outputs=[llm_output],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
