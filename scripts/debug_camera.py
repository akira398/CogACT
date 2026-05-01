"""
debug_camera.py - Render a frame from every camera in the scene and save as PNG.
Run:  python scripts/debug_camera.py
Output: debug_camera_frames/<cam_name>.png  (one image per camera)
        debug_camera_output.txt             (world-frame poses)
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "debug_camera_frames"
OUT_JSON = "debug_camera_output.txt"
TASK = "TurnOnMicrowave"
LAYOUT_ID, STYLE_ID = 1, 1
IMG_SIZE = 256


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

    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect all camera names first with a small env
    print(f"Creating env layout={LAYOUT_ID} style={STYLE_ID} ...")
    env = suite.make(
        TASK, robots="Panda",
        controller_configs=loader(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True,
        use_object_obs=False, use_camera_obs=False,
        camera_names=[], camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
        layout_ids=LAYOUT_ID, style_ids=STYLE_ID,
        obj_instance_split="target", translucent_robot=False,
    )
    env.reset()

    all_cams = [env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)]
    print(f"Found {len(all_cams)} cameras: {all_cams}\n")

    cam_id  = env.sim.model.camera_name2id("robot0_agentview_left") if "robot0_agentview_left" in all_cams else None
    base_id = env.sim.model.body_name2id("robot0_base")

    pose_info = {}
    if cam_id is not None:
        pose_info["agentview_left"] = {
            "cam_xpos": env.sim.data.cam_xpos[cam_id].tolist(),
            "cam_xmat": env.sim.data.cam_xmat[cam_id].reshape(3,3).tolist(),
        }
    pose_info["base"] = {
        "base_xpos": env.sim.data.body_xpos[base_id].tolist(),
        "base_xmat": env.sim.data.body_xmat[base_id].reshape(3,3).tolist(),
    }
    pose_info["all_cameras"] = all_cams
    env.close()

    # Now render each camera individually
    from PIL import Image as PILImage

    results = {}
    for cam_name in all_cams:
        print(f"  Rendering {cam_name} ...", end=" ", flush=True)
        try:
            env2 = suite.make(
                TASK, robots="Panda",
                controller_configs=loader(default_controller="OSC_POSE"),
                has_renderer=False, has_offscreen_renderer=True,
                use_object_obs=False, use_camera_obs=True,
                camera_names=[cam_name],
                camera_heights=IMG_SIZE, camera_widths=IMG_SIZE,
                layout_ids=LAYOUT_ID, style_ids=STYLE_ID,
                obj_instance_split="target", translucent_robot=False,
            )
            obs = env2.reset()
            img = obs[f"{cam_name}_image"]
            if img.ndim == 4:
                img = img[0]
            out_path = os.path.join(OUT_DIR, f"{cam_name}.png")
            PILImage.fromarray(img.astype("uint8")).save(out_path)
            env2.close()
            results[cam_name] = "ok"
            print(f"saved → {out_path}")
        except Exception as e:
            results[cam_name] = str(e)
            print(f"FAILED: {e}")

    pose_info["render_results"] = results
    with open(OUT_JSON, "w") as f:
        json.dump(pose_info, f, indent=2)

    print(f"\nPoses saved to {OUT_JSON}")
    print(f"Images saved to {OUT_DIR}/")
    print("\nCamera list:")
    for cam in all_cams:
        status = results.get(cam, "?")
        print(f"  {cam:40s}  {status}")


if __name__ == "__main__":
    main()
