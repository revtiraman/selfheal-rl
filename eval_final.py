import sys
sys.path.insert(0, '/Users/revtiramantripathi/rl-hackathon')

from stable_baselines3 import PPO
from env.selfheal_env import SelfHealEnv
from core.graders import Grader
from core.heuristic_agent import HeuristicAgent
from config import MAX_STEPS_PER_EPISODE, ACTION_TYPES, SERVICE_NAMES
import numpy as np

final = PPO.load('models/selfheal_agent_final.zip')
N = 20

def run_ppo(diff, seed):
    env = SelfHealEnv(difficulty=diff, partial_observability=True)
    obs, _ = env.reset(seed=seed)
    for _ in range(MAX_STEPS_PER_EPISODE):
        a, _ = final.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc: break
    g = Grader.grade_all(env.get_episode_summary())
    return env.mesh.is_fully_recovered(), env.total_reward, g['overall_score']

def run_heuristic(diff, seed):
    env = SelfHealEnv(difficulty=diff, partial_observability=False)
    obs, _ = env.reset(seed=seed)
    h = HeuristicAgent(); h.reset()
    for _ in range(MAX_STEPS_PER_EPISODE):
        st = env.mesh.get_all_statuses()
        at, tgt = h.act(st)
        if at == 'observe':
            svc = env.mesh.services.get(tgt)
            if svc: h.record_observation(tgt, svc.failure_type or 'unknown')
        action = ACTION_TYPES.index(at) * len(SERVICE_NAMES) + SERVICE_NAMES.index(tgt)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: break
    g = Grader.grade_all(env.get_episode_summary())
    return env.mesh.is_fully_recovered(), env.total_reward, g['overall_score']

print("\n=== SelfHealRL Final Evaluation (N=20, reward-hacking fixed) ===")
print(f"{'':8}  {'PPO':^22}  {'Heuristic':^22}")
print(f"{'Diff':8}  {'Succ':>6} {'Reward':>7} {'Grade':>6}  {'Succ':>6} {'Reward':>7} {'Grade':>6}")
print('-' * 52)

for diff in ['EASY', 'MEDIUM', 'HARD', 'CHAOS']:
    pp = [run_ppo(diff, i) for i in range(N)]
    hh = [run_heuristic(diff, i) for i in range(N)]
    ps = np.mean([r[0] for r in pp]); pr = np.mean([r[1] for r in pp]); pg = np.mean([r[2] for r in pp])
    hs = np.mean([r[0] for r in hh]); hr = np.mean([r[1] for r in hh]); hg = np.mean([r[2] for r in hh])
    print(f"{diff:8}  {ps:>5.0%}  {pr:>+7.1f}  {pg:>6.2f}  {hs:>5.0%}  {hr:>+7.1f}  {hg:>6.2f}")

print()
