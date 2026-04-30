"""
test_setup.py — quick sanity check before running the full eval.

Runs in ~30 seconds and catches the most common failures:
  1. Package imports + version checks
  2. RoboCasa environment creation (one task, one step)
  3. CogACT model load
  4. One end-to-end inference call (image → action)

Usage:
    python scripts/test_setup.py \
        --model_path pretrained/CogACT-Base \
        --action_model_type DiT-B \
        --norm_stats_path data/robocasa/dataset_statistics.json \
        --unnorm_key robocasa

    # Skip model test (env only):
    python scripts/test_setup.py --skip_model

    # Skip env test (model only):
    python scripts/test_setup.py --model_path pretrained/CogACT-Base --skip_env
"""

import argparse
import sys
import traceback


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"


def result(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n── {title} {'─' * (50 - len(title))}")


# ── 1. Imports ────────────────────────────────────────────────────────────────

def test_imports() -> bool:
    section("1. Core imports")
    ok = True

    checks = [
        ("numpy",      lambda: __import__("numpy").__version__,          ">=2.2"),
        ("torch",      lambda: __import__("torch").__version__,           None),
        ("PIL",        lambda: __import__("PIL").__version__,             None),
        ("h5py",       lambda: __import__("h5py").__version__,            None),
        ("robosuite",  lambda: __import__("robosuite").__version__,       ">=1.5"),
        ("robocasa",   lambda: __import__("robocasa").__version__,        None),
    ]

    for name, fn, req in checks:
        try:
            ver = fn()
            detail = f"v{ver}"
            if req:
                import packaging.version as pv
                op, target = req[:2], req[2:]
                v = pv.parse(ver)
                t = pv.parse(target)
                ver_ok = (v >= t) if op == ">=" else True
                detail += f"  {GREEN if ver_ok else RED}(need {req}){RESET}"
                ok &= ver_ok
            result(name, True, detail)
        except Exception as e:
            result(name, False, str(e))
            ok = False

    # Check load_controller_config API
    section("1b. robosuite API compatibility")
    try:
        import robosuite as suite
        has_old = hasattr(suite, "load_controller_config")
        has_new = hasattr(suite, "load_part_controller_config")
        result("load_controller_config",      has_old, "present" if has_old else "missing")
        result("load_part_controller_config", has_new, "present" if has_new else "missing")
        if not has_old and not has_new:
            print(f"    {RED}Neither controller config loader found — env creation will fail{RESET}")
            ok = False
    except Exception as e:
        result("robosuite API check", False, str(e))
        ok = False

    # Check robocasa v1.0 extra params (patched automatically if missing)
    try:
        import inspect
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        from robosuite.models.tasks import Task
        for cls, param in [(ManipulationEnv, "load_model_on_init"), (Task, "enable_multiccd")]:
            has = param in inspect.signature(cls.__init__).parameters
            detail = "native" if has else f"{YELLOW}missing — will be monkey-patched{RESET}"
            result(f"{cls.__name__}.{param}", True, detail)
    except Exception as e:
        result("robosuite compat check", False, str(e))
        ok = False

    return ok


# ── 2. Environment creation ───────────────────────────────────────────────────

def _patch_robosuite_compat() -> None:
    """Patch robosuite 1.5.x kwargs missing vs robocasa v1.0 expectations."""
    import inspect

    def _add_kwarg(cls, param: str) -> None:
        if param in inspect.signature(cls.__init__).parameters:
            return
        _orig = cls.__init__
        def _patched(self, *args, **kwargs):
            kwargs.pop(param, None)
            return _orig(self, *args, **kwargs)
        cls.__init__ = _patched

    try:
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        _add_kwarg(ManipulationEnv, "load_model_on_init")
    except Exception:
        pass

    try:
        from robosuite.models.tasks import Task
        _add_kwarg(Task, "enable_multiccd")
    except Exception:
        pass


def test_env(task: str = "TurnOnMicrowave") -> bool:
    section(f"2. RoboCasa environment ({task})")

    try:
        import robosuite as suite
        try:
            import robocasa.environments  # noqa
        except ImportError:
            import robocasa  # noqa

        loader = getattr(suite, "load_controller_config",
                         getattr(suite, "load_part_controller_config", None))
        if loader is None:
            result("controller config loader", False, "not found in robosuite")
            return False

        _patch_robosuite_compat()
        result("robosuite + robocasa import", True)
    except Exception as e:
        result("robosuite + robocasa import", False, str(e))
        return False

    try:
        env = suite.make(
            env_name=task,
            robots="Panda",
            controller_configs=loader(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_object_obs=False,
            use_camera_obs=True,
            camera_names=["robot0_agentview_left"],
            camera_heights=128,
            camera_widths=128,
            layout_ids=1,
            style_ids=1,
            obj_instance_split="B",
            translucent_robot=False,
        )
        result("env creation", True)
    except Exception as e:
        result("env creation", False, str(e))
        traceback.print_exc()
        return False

    try:
        obs = env.reset()
        img = obs["robot0_agentview_left_image"]
        result("env.reset()", True, f"image shape={img.shape}")
    except Exception as e:
        result("env.reset()", False, str(e))
        env.close()
        return False

    try:
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        result("env.step(random_action)", True)
    except Exception as e:
        result("env.step()", False, str(e))
        env.close()
        return False

    env.close()
    result("env.close()", True)
    return True


# ── 3. Model load ─────────────────────────────────────────────────────────────

def test_model(args) -> bool:
    section("3. CogACT model load")

    try:
        from vla import load_vla
        result("vla import", True)
    except Exception as e:
        result("vla import", False, str(e))
        return False

    import json
    from pathlib import Path

    try:
        p = Path(args.model_path)
        if p.is_dir():
            ckpts = sorted((p / "checkpoints").glob("*.pt"))
            assert ckpts, f"No .pt in {p / 'checkpoints'}"
            ckpt = str(ckpts[-1])
        else:
            ckpt = args.model_path
        result("checkpoint found", True, ckpt)
    except Exception as e:
        result("checkpoint found", False, str(e))
        return False

    try:
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = load_vla(
            ckpt,
            load_for_training=False,
            action_model_type=args.action_model_type,
            action_dim=7,
            future_action_window_size=15,
            past_action_window_size=0,
            use_ema=False,
        )
        result("load_vla", True)
    except Exception as e:
        result("load_vla", False, str(e))
        traceback.print_exc()
        return False

    if args.norm_stats_path:
        try:
            with open(args.norm_stats_path) as f:
                stats = json.load(f)
            if model.norm_stats is None:
                model.norm_stats = {}
            model.norm_stats.update(stats)
            result("norm_stats patch", True, f"keys={list(stats.keys())}")
        except Exception as e:
            result("norm_stats patch", False, str(e))

    return model


# ── 4. End-to-end inference ───────────────────────────────────────────────────

def test_inference(model, unnorm_key: str) -> bool:
    section("4. End-to-end inference (dummy image)")
    import torch
    import numpy as np
    from PIL import Image as PILImage

    try:
        device = next(model.parameters()).device
        dummy_img = PILImage.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        with torch.no_grad():
            actions, _ = model.predict_action(
                image=dummy_img,
                instruction="pick up the cup",
                unnorm_key=unnorm_key,
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
                do_sample=False,
            )
        result("predict_action", True, f"output shape={actions.shape}  device={device}")
        return True
    except Exception as e:
        result("predict_action", False, str(e))
        traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--action_model_type", type=str, default="DiT-B",
                   choices=["DiT-S", "DiT-B", "DiT-L"])
    p.add_argument("--norm_stats_path", type=str, default=None)
    p.add_argument("--unnorm_key", type=str, default="robocasa")
    p.add_argument("--test_task", type=str, default="TurnOnMicrowave",
                   help="RoboCasa task to test env creation with.")
    p.add_argument("--skip_env", action="store_true")
    p.add_argument("--skip_model", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 54)
    print("  CogACT + RoboCasa setup verification")
    print("=" * 54)

    results = {}

    results["imports"] = test_imports()

    if not args.skip_env:
        results["env"] = test_env(args.test_task)
    else:
        print(f"\n── 2. RoboCasa environment {'─' * 30}")
        print(f"  [{SKIP}] env test (--skip_env)")

    model = None
    if not args.skip_model:
        if args.model_path is None:
            print(f"\n── 3. CogACT model load {'─' * 31}")
            print(f"  [{SKIP}] no --model_path given")
        else:
            model = test_model(args)
            results["model"] = bool(model)

            if model and args.unnorm_key:
                results["inference"] = test_inference(model, args.unnorm_key)

    print("\n" + "=" * 54)
    print("  Summary")
    print("=" * 54)
    all_ok = True
    for name, ok in results.items():
        print(f"  {name:<12} {'✓' if ok else '✗'}  {'OK' if ok else 'FAILED'}")
        all_ok &= ok

    if all_ok:
        print(f"\n  {GREEN}All checks passed — ready to run eval.{RESET}")
    else:
        print(f"\n  {RED}Some checks failed — fix issues above before running eval.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
