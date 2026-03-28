"""SelfHealRL — Entry point for training, evaluation, and demo."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="SelfHealRL — Autonomous Microservices Recovery")
    sub = parser.add_subparsers(dest="command")

    # Train
    train_p = sub.add_parser("train", help="Train the RL agent")
    train_p.add_argument("--phase", default=None, help="Training phase (phase1_easy, phase2_medium, ...)")
    train_p.add_argument("--curriculum", action="store_true", help="Run full curriculum training")

    # Evaluate
    eval_p = sub.add_parser("eval", help="Evaluate a trained model")
    eval_p.add_argument("--model", default="models/selfheal_agent_final.zip", help="Path to model")
    eval_p.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    eval_p.add_argument("--difficulty", default="MEDIUM", help="Difficulty level")

    # Demo
    demo_p = sub.add_parser("demo", help="Launch Gradio web interface")
    demo_p.add_argument("--share", action="store_true", help="Create public share link")

    # Quick test
    sub.add_parser("test", help="Run a quick environment test")

    args = parser.parse_args()

    if args.command == "train":
        from training.train import Trainer
        trainer = Trainer()
        if args.curriculum:
            trainer.train_curriculum()
        elif args.phase:
            trainer.train(args.phase)
        else:
            print("Specify --phase or --curriculum")

    elif args.command == "eval":
        from training.train import Trainer
        trainer = Trainer()
        stats = trainer.evaluate(args.model, args.episodes, args.difficulty)
        print(f"\nResults ({args.episodes} episodes, {args.difficulty}):")
        print(f"  Success rate: {stats['success_rate']:.0%}")
        print(f"  Mean reward:  {stats['mean_reward']:.1f}")
        print(f"  Mean grade:   {stats['mean_grade_score']:.2f}")

        comparison = trainer.compare_with_baseline(args.model, args.episodes, args.difficulty)
        t = comparison["trained"]
        r = comparison["random"]
        print(f"\n  Trained: {t['success_rate']:.0%} success, {t['mean_reward']:.1f} reward")
        print(f"  Random:  {r['success_rate']:.0%} success, {r['mean_reward']:.1f} reward")

    elif args.command == "demo":
        from ui.app import build_app
        app = build_app()
        app.launch(share=args.share)

    elif args.command == "test":
        from env.selfheal_env import SelfHealEnv
        from core.graders import Grader

        print("Running quick environment test...")
        env = SelfHealEnv(difficulty="EASY", partial_observability=False)

        try:
            from stable_baselines3.common.env_checker import check_env
            check_env(env)
            print("✅ SB3 environment check passed")
        except ImportError:
            print("⚠️  stable-baselines3 not installed, skipping check")

        obs, info = env.reset(seed=42)
        total_reward = 0
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            if term or trunc:
                break

        summary = env.get_episode_summary()
        grades = Grader.grade_all(summary)

        print(f"✅ Episode completed: {step+1} steps, reward={total_reward:.1f}")
        print(f"   Recovered: {env.mesh.is_fully_recovered()}")
        print(f"   Grade: {grades['overall_score']:.2f}")
        print("✅ All systems operational!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
