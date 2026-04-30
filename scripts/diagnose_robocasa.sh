#!/usr/bin/env bash
# diagnose_robocasa.sh
# Runs all robocasa setup diagnostics and saves output to robocasa_diag.txt
# Usage: bash scripts/diagnose_robocasa.sh

OUT="robocasa_diag.txt"
exec > >(tee "$OUT") 2>&1

echo "======================================================"
echo "  RoboCasa diagnostics — $(date)"
echo "======================================================"

# ── Python / conda env ────────────────────────────────────
echo ""
echo "── Python environment ────────────────────────────────"
which python
python --version
echo "Conda env: ${CONDA_DEFAULT_ENV:-not set}"

# ── Key package versions ──────────────────────────────────
echo ""
echo "── Package versions ──────────────────────────────────"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import mujoco; print('mujoco:', mujoco.__version__)"
python -c "import robosuite; print('robosuite:', robosuite.__version__)" 2>/dev/null || echo "robosuite: no __version__"
python -c "import robocasa; print('robocasa:', robocasa.__version__)" 2>/dev/null || echo "robocasa: import failed"

# ── robocasa install location ─────────────────────────────
echo ""
echo "── robocasa install location ─────────────────────────"
python -c "import robocasa; print(robocasa.__file__)"

# ── macros_private.py ─────────────────────────────────────
echo ""
echo "── macros_private.py ─────────────────────────────────"
MACROS_DIR=$(python -c "import os, robocasa; print(os.path.dirname(robocasa.__file__))" 2>/dev/null)
echo "robocasa dir: $MACROS_DIR"

MACROS_FILE="$MACROS_DIR/macros_private.py"
if [ -f "$MACROS_FILE" ]; then
    echo "macros_private.py: EXISTS"
    echo "--- contents ---"
    cat "$MACROS_FILE"
    echo "----------------"
else
    echo "macros_private.py: MISSING"
    echo "Need to run: python $MACROS_DIR/scripts/setup_macros.py"
fi

# ── DATASET_BASE_PATH ─────────────────────────────────────
echo ""
echo "── DATASET_BASE_PATH ─────────────────────────────────"
python -c "
import robocasa.macros as m
print('DATASET_BASE_PATH:', m.DATASET_BASE_PATH)
import os
if m.DATASET_BASE_PATH:
    exists = os.path.exists(m.DATASET_BASE_PATH)
    print('exists:', exists)
    if exists:
        contents = os.listdir(m.DATASET_BASE_PATH)
        print('contents:', contents[:10])
else:
    print('NOT SET — run setup_macros.py')
" 2>/dev/null || echo "Could not read DATASET_BASE_PATH"

# ── Kitchen assets ────────────────────────────────────────
echo ""
echo "── Kitchen assets (robocasa package) ─────────────────"
ASSETS_DIR=$(python -c "
import os, robocasa
base = os.path.dirname(robocasa.__file__)
for sub in ['models/assets', 'models/objects/assets', 'assets']:
    p = os.path.join(base, sub)
    if os.path.exists(p):
        print(p)
        break
else:
    print('NOT FOUND')
" 2>/dev/null)
echo "assets dir: $ASSETS_DIR"
if [ -d "$ASSETS_DIR" ]; then
    echo "top-level contents:"
    ls "$ASSETS_DIR" | head -20
    echo "total files: $(find "$ASSETS_DIR" -type f | wc -l)"
fi

# ── OBJ_CATEGORIES check ──────────────────────────────────
echo ""
echo "── OBJ_CATEGORIES (registered kitchen objects) ───────"
python -c "
try:
    from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
    cats = list(OBJ_CATEGORIES.keys())
    print(f'categories found: {len(cats)}')
    print('first 5:', cats[:5])
except Exception as e:
    print('ERROR:', e)
" 2>/dev/null

# ── Quick env creation test ───────────────────────────────
echo ""
echo "── Quick env creation test ───────────────────────────"
python -c "
import sys, inspect
sys.path.insert(0, '.')

def _make_permissive(cls):
    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    valid = frozenset(sig.parameters.keys()) - {'self'}
    orig = cls.__init__
    def patched(self, *args, **kwargs):
        for k in list(kwargs):
            if k not in valid: kwargs.pop(k)
        return orig(self, *args, **kwargs)
    cls.__init__ = patched

try:
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    _make_permissive(ManipulationEnv)
    from robosuite.models.tasks import Task
    _make_permissive(Task)
except Exception as e:
    print('patch error:', e)

import robosuite as suite
import robocasa.environments
loader = getattr(suite, 'load_controller_config', getattr(suite, 'load_part_controller_config', None))

# Monkey-patch sample_kitchen_object_helper to show what 'groups' value causes the error
from robocasa.models.objects import kitchen_object_utils as kou
_orig_helper = kou.sample_kitchen_object_helper
def _debug_helper(groups, *a, **kw):
    try:
        return _orig_helper(groups, *a, **kw)
    except ValueError:
        print(f'  [DEBUG] sample_kitchen_object_helper failed for groups={repr(groups)}, split={kw.get(\"obj_instance_split\", kw.get(\"split\", \"?\"))}')
        raise
kou.sample_kitchen_object_helper = _debug_helper

for split in ['A', 'B']:
    for task in ['TurnOnMicrowave', 'OpenDrawer', 'CloseFridge']:
        try:
            env = suite.make(
                env_name=task, robots='Panda',
                controller_configs=loader(default_controller='OSC_POSE'),
                has_renderer=False, has_offscreen_renderer=True,
                use_object_obs=False, use_camera_obs=True,
                camera_names=['robot0_agentview_left'],
                camera_heights=128, camera_widths=128,
                layout_ids=1, style_ids=1,
                obj_instance_split=split, translucent_robot=False,
            )
            obs = env.reset()
            print(f'  PASS  task={task}  split={split}  image={obs[\"robot0_agentview_left_image\"].shape}')
            env.close()
            break  # one success is enough
        except ValueError:
            print(f'  FAIL  task={task}  split={split}  → ValueError (see DEBUG above)')
        except Exception as e:
            print(f'  FAIL  task={task}  split={split}  → {type(e).__name__}: {e}')
" 2>&1

echo ""
echo "======================================================"
echo "  Saved to: $OUT"
echo "======================================================"
