# Original HoST → MJLab audit

## Scope and provenance

The executable source of truth is HoST commit `70bb580949a336a920833700e4b5dc3bf7fe87ce`, specifically `legged_gym/envs/pi/pi_host_ground.py`, `pi_config_ground.py`, inherited `base/legged_robot_config.py`, the original `pi_12dof_release_v1.urdf`, and `rsl_rl/rsl_rl/{modules,storage,algorithms}`. The task, algorithm and MJCF structure originate from HoST. At the user’s subsequent request, all 13 meshes referenced by the MJCF were replaced with copies from `mjlab_playground/src/mjlab_playground/asset_zoo/robots/hightorque_minipi/xmls/assets/`. No runtime dependency on that repository is introduced. Both input repositories were read only.

The reference project's `pyproject.toml`, getup base configuration, T1 registration, MDP organization and workflow were inspected. Its virtual environment initially contained no MJLab installation. API inspection first used the adjacent MJLab checkout (commit `5ae450e88067365c8914acc8a6717b62a85f4f66`), then the actual installed `mjlab==1.6.0` code. This project pins a released, independently installable stack rather than inheriting the reference's git/nightly overrides. The lockfile contains no local source-repository dependency.

Only the original **Mini-Pi ground** task is registered. H1/G1, slope, wall, platform and prone-specific task variants have not been ported. The Mini-Pi ground source itself contains comments describing earlier tuning; the checked-in executable equations/configuration take precedence over those comments.

## Component mapping

| Original HoST component | Native MJLab implementation |
| --- | --- |
| `BaseTask` simulation loop and scene | Unmodified `ManagerBasedRlEnv`, `SceneCfg`, `SimulationCfg`, MuJoCo-Warp |
| `compute_observations` | `ObservationTermCfg` calling `mdp.observations.history`; six-frame state retained in `HostState` |
| `_compute_torques` and decimation loop | Custom `ActionTerm` processing once/control step, applying torques every physics substep through `BuiltinMotorActuatorCfg` |
| `compute_reward` | Four `RewardTermCfg` terms; task product and three additive groups, separately exposed by adapter |
| `check_termination` | Timeout and velocity `TerminationTermCfg` terms |
| `_reset_dofs`, `_reset_root_states` | Reset `EventTermCfg`, public entity state writers |
| Shape/body creation randomization | Startup `EventTermCfg` declaring expanded model fields and constant recomputation |
| Gain/strength/offset/delay resampling | Reset event |
| `update_force_curriculum` | `CurriculumTermCfg`, called on terminal state before reset |
| Contact tensor | Native body `ContactSensorCfg`, net force reduction |
| URDF asset loader | Reproducible URDF → MJCF conversion, native MJLab `EntityCfg` |
| Original VecEnv return tuple | `HostAdapter`: four-group reward, terminated OR timeout, timeout mask, actor/optional critic observations |
| Original runner | Small local logging/checkpoint runner around copied HoST PPO; no single-critic substitution |

The custom shared state owns task buffers and state views, not simulation. No environment subclass or MJLab core change is needed. The first termination term performs a `sim.forward()` before rewards to avoid MJLab's usual one-substep stale derived poses.

## Exact task/algorithm semantics retained

- Physics step 0.005 s, decimation 4, control step 0.02 s, gravity `(0,0,-9.81)`. Timeout is strictly `episode_length > 500`, so an unmodified episode expires on step 501 (10.02 s).
- The separate real-step counter gates actions and observations at **strictly greater than 30**. Observation frame at real step 31 is visible; the first nonzero commanded action can be applied on the following control step. Zero actions do not imply zero torques: damping and randomized offsets remain active.
- Twelve URDF joints are explicitly mapped by name in URDF tree order (right leg then left leg). Torque/velocity/range limits come from the same URDF. The original Isaac Gym runtime joint enumeration could not be measured here; cross-engine checkpoint joint ordering must be checked before reusing an external trained checkpoint.
- Actor observation order: angular velocity × .25 (3), projected gravity (3), absolute joint positions (12), joint velocity × .05 (12), processed actions (12), action rescale plus U(-.025,.025) (1). Uniform additive noise amplitudes are .05, .05, .01, .075, 0 and 0 respectively. Six frames = 258 values. Clip observations and actions at ±100.
- Observation history survives resets and shifts in zero frames during settling. Native automatic observation-history clearing is intentionally not used. Original privileged observations are `None`; the critic uses the same noisy actor history. Adapter accepts a `critic` group if supplied, but changing its dimensionality would require revisiting the original smoothness loss, which evaluates critics on mixed actor observations.
- Position targets are **current joint position + delayed scaled action**, recomputed every 5 ms. Delay buffer length 5; sampled index 0–4 corresponds to 4–0 substeps of latency (20–0 ms), not policy steps. Gains are hip pitch/calf 30, hip roll/thigh 15, ankle pitch 12, ankle roll 5; damping .2 on every joint. Multiplicative Kp/Kd U(.85,1.15), strength U(.9,1.1), additive torque offset U(-.05,.05) × effort limit, then hard torque clamp.
- Task reward = orientation tolerance × head-height tolerance, without dt scaling. `regu`, `style`, `target` use original equations/scales × .02. MJLab's extra dt scaling is disabled. The scalar MJLab reward is only a diagnostic sum; PPO receives `[task, regu, style, target]`.
- Head-height reward subtracts mean foot-link height. The ground-parallel reward includes the four auxiliary bodies plus ankle-roll body on each foot, retaining unbiased `var()` over five points.
- The cohort mean of the most recent relative head height must exceed .37 m before reducing pull force by 20 N and action rescale by .02 (floors 0 N and .25). Initial force is 15 N: the first qualifying update removes it entirely. This is not a staged recovery curriculum.
- Force goes to the base body, world +Z, only after settling and while projected gravity z < -.8. A pending-force buffer preserves the source's post-simulation submission timing (one physics substep). Isaac Gym's default force coordinate space is ENV_SPACE; see the [NVIDIA-authored Tensor API documentation mirror](https://junxnone.github.io/isaacgymdocs/programming/tensors.html#applying-forces).
- The adapter performs one zero-action step after global reset, matching `BaseTask.reset()`. Automatic per-episode resets occur within the normal manager step and do not add a simulation step.
- Reset joint positions U(-.1,.1), clipped to .9 × URDF limits; velocities zero; base position `(0,0,.351)`, fixed initial orientation. No added random fall distribution.
- The source's push call is commented out. This implementation also does not push, despite `push_robots=True` in the config.
- Source link-mass randomization overwrites its earlier payload assignment (the base branch contains `pass`, not `continue`). Effective masses therefore follow default mass × U(.8,1.2), including the base. COM displacement is U(-.03,.03) m per axis on the base. No extra additive payload is introduced.
- Velocity termination retains `abs(max(joint_velocity)) > 300`, not `max(abs(...))`, plus base-speed norm >20 m/s, both gated after settling. Configured terminating-contact list is empty.
- Actor 258→512→256→128→12, ELU hidden activations, final tanh; four independent 258→512→256→1 ELU critics; learned Gaussian std initially .8.
- Group GAE and per-group sample-standard-deviation normalization, aggregated with weights `[2.5,.1,1,1]`. Gamma .99, lambda .95, five epochs, four minibatches, clipped objective/value loss .2, entropy .01, Adam lr .001, adaptive KL .01, grad norm 1. Original policy/value interpolation smoothness retained.
- Original storage drops the last rollout time step when forming minibatches, reuses its random permutation across epochs, and PPO skips transitions whose whole-batch observation norm is ≤1e-4. The runner still collects exactly 50 environment steps/update. Consequently the first rollout may contain fewer stored transitions, leaving zero slots used by original GAE. This is logged as `Rollout/stored_steps`; it is not silently corrected.
- Timeout bootstrapping deliberately uses the **pre-step** four-critic value (`reward += gamma * V(s_t) * timeout`), as original PPO does, followed by done-masked GAE. Substituting V(terminal observation) would change the original algorithm. Terminal observation snapshots are provided for diagnostics; they do not replace the original bootstrap. Snapshot noise does not advance the training RNG stream.

`original_config.py` resolves inherited settings without importing legged_gym. Original algorithm and reward function bodies retain their source equations; package imports are local.

## Physics differences and quantified bounds

Exact trajectory equality is impossible across PhysX and MuJoCo. These are explicit implementation choices, not claims of empirical equivalence:

| Difference | Port behavior and quantitative impact |
| --- | --- |
| Non-unit initial quaternion | Original xyzw `(0,-1,0,1)` has norm √2. MuJoCo uses normalized wxyz `(1/√2,0,-1/√2,0)`, same represented rotation. Isaac's normalization behavior was not measured. |
| Solver | PhysX TGS 8 position + 1 velocity iteration becomes MuJoCo Newton 8 iterations with Euler integration. Iteration meanings differ; constraint errors are not numerically equivalent. |
| Contact geometry | Collision primitives/mesh files, full inertias and fixed auxiliary bodies retained; URDF cylinders become capsules as requested by original Isaac asset options. Each engine's convex hull/collision manifold construction can differ. MuJoCo's parent/welded-body filtering also differs from Isaac's actor filtering. |
| Contact offset | Robot and explicit ground pairs use margin .01 m and gap .01 m, giving speculative detection at 1 cm with zero-distance active constraints. This approximates PhysX contact_offset=.01/rest_offset=0; it is not the same solver treatment. MuJoCo-Warp native/multicontact CCD flags are disabled because that backend rejects nonzero box-box margins with those flags enabled. Contact manifold counts may therefore differ. |
| Friction | Per-robot coefficient U(.1,1) is retained. Explicit robot-ground pairs use average material mixing with static ground coefficient .8, yielding .45–.9. MuJoCo cannot switch to the source's separate dynamic ground coefficient .7: during sliding the retained coefficient is .05 higher than an averaged dynamic coefficient. Self contacts use the sampled robot coefficient. Rolling/torsional coefficients are zero. |
| Restitution | Source samples U(0,1), with ground .3. Self contacts use the sample; ground pairs use its average with .3. Map effective e to damping ratio ζ = -ln(e)/√(π²+ln(e)²), at time constant .02 s. This ideal linear-impact mapping is an approximation, not a calibrated restitution guarantee. Source bounce threshold .5 m/s and max depenetration velocity 1 m/s have no equivalent implemented MuJoCo setting. |
| Mass/inertia randomization | Mass ratios are exact .8–1.2; MuJoCo diagonal inertia is scaled by the same ratio. PhysX `recomputeInertia=True` can recompute from collision geometry instead of scaling URDF inertia. COM offsets match ±3 cm, but recomputed inertia can therefore differ; no Isaac measurements were available to quantify the tensor discrepancy. |
| Body damping | Original linear/angular decay .01/s is applied as per-body force -m*.01*v and principal-frame torque -I*.01*ω. Free-body per-step decay is approximately .99995 at dt=.005. Constraint coupling/integration and source velocity caps (1000) are not exactly reproduced. |
| Contacts | Native net-force sensors expose current MuJoCo solved contacts. Original collection mode=2 aggregates substeps; with configured one simulation substep this is the same sampling frequency, not guaranteed identical impulses. None of the active Mini-Pi reward terms reads contact force. |
| State freshness | An extra forward pass refreshes poses before termination/reward evaluation, eliminating MJLab's 5 ms pose lag. Physics integrator/COM velocity definitions still differ between engines. |
| RNG | The distributions and update cadence are retained. Torch versus original NumPy/Isaac sampling, different draw ordering and physics prevent cross-engine bitwise trajectories. Same-seed CPU Warp determinism is tested separately. |

Nominal total robot mass is 6.92121732 kg, with 22 bodies excluding world and 12 actuated joints. Joint limits are stored in `assets/minipi/joints.py`. No Isaac Gym trajectory comparison or empirical restitution calibration has been performed.

## Mesh source update

The 13 active STL files now come from the reference project asset zoo at the user’s request. Joint transforms, masses, inertias, actuators and primitive collision definitions remain those of the HoST-derived MJCF. The hip-pitch mesh is also used for collision on each side, so changing that mesh can change contact geometry. No simulation or training was run for this update.

Files with different bytes from the previously bundled HoST meshes: none (all 13 are byte-identical).
