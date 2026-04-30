"""
debug_camera.py - Print world-frame camera and robot body info for fixed Panda.
Run:  python scripts/debug_camera.py
Output saved to: debug_camera_output.txt
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "debug_camera_output.txt"

def main():
    from scripts.eval_robocasa365 import _patch_robosuite_compat
    _patch_robosuite_compat()

    import robosuite as suite
    try:
        import robocasa.environments
    except ImportError:
        import robocasa

    loader = getattr(suite, "load_part_controller_config",
                     getattr(suite, "load_controller_config", None))

    results = {}

    for layout_id, style_id in [(1, 1), (2, 2)]:
        key = f"layout{layout_id}_style{style_id}"
        print(f"Creating env layout={layout_id} style={style_id} ...")
        try:
            env = suite.make(
                "TurnOnMicrowave", robots="Panda",
                controller_configs=loader(default_controller="OSC_POSE"),
                has_renderer=False, has_offscreen_renderer=True,
                use_object_obs=False, use_camera_obs=True,
                camera_names=["robot0_agentview_left"],
                camera_heights=256, camera_widths=256,
                layout_ids=layout_id, style_ids=style_id,
                obj_instance_split="target", translucent_robot=False,
            )
            env.reset()

            cam_id  = env.sim.model.camera_name2id("robot0_agentview_left")
            base_id = env.sim.model.body_name2id("robot0_base")

            cam_xpos = env.sim.data.cam_xpos[cam_id].tolist()
            cam_xmat = env.sim.data.cam_xmat[cam_id].reshape(3, 3).tolist()
            base_xpos = env.sim.data.body_xpos[base_id].tolist()
            base_xmat = env.sim.data.body_xmat[base_id].reshape(3, 3).tolist()

            # also list all cameras in the model
            all_cams = [env.sim.model.camera_id2name(i)
                        for i in range(env.sim.model.ncam)]

            results[key] = {
                "cam_xpos": cam_xpos,
                "cam_xmat": cam_xmat,
                "base_xpos": base_xpos,
                "base_xmat": base_xmat,
                "all_cameras": all_cams,
            }
            print(f"  cam_xpos:  {cam_xpos}")
            print(f"  base_xpos: {base_xpos}")
            env.close()
        except Exception as e:
            import traceback
            results[key] = {"error": str(e), "traceback": traceback.format_exc()}
            print(f"  ERROR: {e}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUT}")

if __name__ == "__main__":
    main()
