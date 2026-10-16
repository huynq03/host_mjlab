# Static review of HoST MJLab port

This revision was inspected statically only. No project modules were imported,
no environment was instantiated, and no simulation, training, evaluation, smoke
test, pytest, or GPU workload was run. No new test scripts were created.
Earlier execution claims do not validate this revision.

## Changes

### Reset, history and curriculum

- Reset clears the selected environments' pending pull, recorded torque and
  target, simulated external wrenches, and joint effort targets. A pending
  one-substep pull from an episode cannot enter its successor.
- Joint mapping is converted to a device-local `long` tensor. MJLab entity
  target setters only form the outer product with reset environment IDs when
  joint IDs are tensors; passing the list returned by `find_joints` could
  pair indices elementwise or fail on partial reset.
- History advances once per control step for all environments. An explicit
  partial reset in the same step only shifts the selected rows. Repeated
  observation reads do not advance history. Original history carry-over,
  zero frames during settling, frame order, noise and clipping are retained.
- Long-lived action, delay, pending-force, history and head-height buffers are
  updated in place. This avoids replacing normal buffers with tensors allocated
  under the runner's inference mode that could later reject writes during an
  explicit reset outside inference mode.
- Terminal observation snapshots moved to the step event, before MJLab resets
  simulator/entity buffers. Sampling them uses a forked CPU/current CUDA RNG;
  it does not advance the policy's training random stream. These snapshots are
  diagnostics, not an alternative timeout-bootstrap rule.
- Installed MJLab orders each step as termination → rewards → step events →
  `_reset_idx`. `_reset_idx` runs curriculum → simulator reset → scene reset →
  reset events → manager resets. The curriculum therefore sees terminal
  relative head height before the reset event clears it. No second curriculum
  update was added. Threshold .37 m, cohort averaging, -20 N, -.02 and floors
  0 N/.25 remain unchanged.

### Independently configurable native rewards

There are 24 original active manager entries: two task, nine regularization,
eight style and five target terms. Names and baseline coefficients come from
`original_config.py`; each entry exposes its own `RewardTermCfg.weight`.
No new recovery reward or stage was introduced.

Each function delegates to the original equation and caches its unweighted
result. `RewardManager` logs each weighted term. The `aggregate_rewards` step
event reads the public `active_terms` and `get_term_cfg()` APIs and reduces:

- `task = product(raw_term * coefficient)`;
- each other group `= sum(raw_term * (coefficient * .02))`.

The original factor/addition order and in-place float32 group accumulation are
retained. Zero weights/removed entries are omitted, including from the task
product. MJLab's global reward dt scaling stays disabled. Its scalar reward is
a diagnostic sum of individual weighted terms, not HoST's four-group reward.
`HostAdapter` verifies that aggregation occurred on the current step before
returning the four groups to PPO. Group reduction does not silently adopt
MJLab's NaN sanitization: original HoST numerical behavior is retained.

The head-height equation still runs once to track physical curriculum progress
if its reward is explicitly disabled. This affects that nonbaseline reward
ablation only; baseline coefficients, equations, gates and curriculum are
unchanged. Head-height state assignments now copy into persistent buffers;
the equations and previous-height comparison are unchanged.

### Checkpoints and runner compatibility

- Existing HoST `model_state_dict`/`optimizer_state_dict` checkpoints remain
  supported. Older files without explicit `learning_rate` restore the rate
  from saved Adam parameter groups. New files restore their explicit rate to
  both `PPO.learning_rate` and every optimizer group. This avoids the original
  load path resetting adaptive KL scheduling to the constructor's .001.
- `load` accepts MJLab 1.6.0 playback's `load_cfg`, `strict`, and `map_location`
  keywords as well as this project's `load_optimizer` keyword. Actor-only
  loading includes actor weights and learned standard deviation, while leaving
  optimizer state, critics and update count untouched. Full resume restores
  all model/optimizer state and update count.
- Inference accepts either a tensor or MJLab's TensorDict with an `actor` key.
  The native training hook `add_git_repo_to_log` is provided. No logging
  directory is created when playback supplies `log_dir=None`.
- Native MJLab's scalar RSL wrapper is unwrapped for HoST training. The adapter
  requires auto-reset and 258-dimensional actor/critic observations; optional
  same-sized critic observations get their own rollout storage. The original
  critic smoothness loss still uses mixed actor observations, so arbitrary
  privileged dimensions are deliberately rejected rather than changing PPO.
- Supplied registered policy/algorithm configuration is honored, with original
  defaults, 50 rollout steps and advantage weights `[2.5, .1, 1, 1]`. TensorBoard
  remains the original logger. Too-small minibatch configurations and
  mismatched policy/environment devices fail explicitly. Distributed training
  is rejected because original HoST has no gradient synchronization.
- CLI closes its writer in `finally`, including on checkpoint-load failures.

`actor_critic.py`, `ppo.py`, and `rollout_storage.py` have no changes in this
revision. Group-wise GAE, normalized advantage aggregation, clipping,
optimization, timeout bootstrapping, zero-observation skipping, and omission of
the last rollout sample are unchanged. Robot assets, joint limits, baseline
settings and physics configuration also have no changes.

## Installed API source inspected

All paths below are under `.venv/lib/python3.13/site-packages/`. Package metadata
identifies MJLab **1.6.0**. Source files were read directly, without importing
them or substituting interfaces from the adjacent source repositories.

| Source | Contract checked |
| --- | --- |
| `mjlab/envs/manager_based_rl_env.py` | Initialization order, five-value step, reward/step-event/reset ordering, observation update, reset return, auto-reset |
| `mjlab/managers/action_manager.py` | `build(env)`, `action_dim`, `raw_action`, `process_actions`, per-substep `apply_actions`, reset |
| `mjlab/managers/observation_manager.py` | Term functions receive env and params, not reset env IDs; cache, shape inference and reset behavior |
| `mjlab/managers/reward_manager.py` | `RewardTermCfg`, coefficient/dt scaling, zero-weight skipping, public term configuration, scalar sanitization |
| `mjlab/managers/event_manager.py` | Startup/reset/step signatures, declared model expansion, constant recomputation |
| `mjlab/managers/termination_manager.py` | Boolean `(num_envs,)` terms, timeout/termination masks survive manager reset |
| `mjlab/managers/manager_base.py` | Config resolution, callable/class construction, shape contract |
| `mjlab/entity/entity.py`, `mjlab/entity/data.py` | Joint/body mapping, entity reset and effort/wrench writer signatures, outer indexing |
| `mjlab/actuator/builtin_actuator.py` | Motor consumes effort targets; effort bounds are retained |
| `mjlab/tasks/registry.py` | Registered custom runner and copied train/play configurations |
| `mjlab/rl/vecenv_wrapper.py` | Wrapper reset, TensorDict observations, scalar rewards and `unwrapped` |
| `mjlab/scripts/train.py`, `mjlab/scripts/play.py` | Runner constructor, git logging hook, learn, checkpoint load arguments, inference callable |
| `rsl_rl/runners/on_policy_runner.py`, `rsl_rl/algorithms/ppo.py` | Native runner checkpoint-selection conventions, used for API compatibility only |

## Static checks and remaining risks

AST parsing succeeded for 31 Python files. Ruff lint passed for `src`,
format checking passed for the 12 modified Python files, and `git diff --check`
passed. The inherited generated `joints.py` and `original_config.py` were left
byte-identical rather than reformatted. These inspect syntax/style without
executing the project. Git diff inspection confirms the three core HoST algorithm files
and baseline configuration/physics files are unchanged. These checks cannot
establish runtime or numerical correctness.

Remaining limits:

- CUDA kernel capture, model-field expansion, simultaneous per-world resets,
  contact sensors, full optimizer resume and viewer integration need eventual
  runtime verification on the target machine. None was run for this review.
- PhysX/MuJoCo collision, restitution, inertia recomputation and damping
  differences remain as documented in `AUDIT.md`; this revision does not tune
  them or claim equivalent trajectories. Motor limits and physics settings
  were not changed.
- Checkpoints do not contain simulator state, per-environment assistance,
  action histories or random-generator states, matching the original save
  scope. Resume restores optimization but is not bitwise continuation of a
  trajectory; environment reset semantics still apply.
- Original PPO's partially filled first rollout, direct learned Gaussian std,
  critic smoothness assumptions, and pre-step timeout value bootstrap remain
  baseline limitations. The unused original `PPO.test_mode()` helper calls a
  nonexistent `.test()` method; the supported runner uses `policy.eval()`
  instead so that helper is not invoked. It is left unchanged with PPO.
- Individual terms are editable by weight or removal. Custom replacement
  functions must participate in the `_evaluate` raw-value cache contract;
  adding unrelated reward functions without it is unsupported.
