# H-LeWM: Hierarchical Latent World Model Methodology

## Big idea

H-LeWM keeps the original LeWM as the **low-level short-horizon world model** and adds a new **high-level long-horizon world model** on top.

In simple terms:

```text
LeWM:
Given current image + primitive action,
predict the next image's latent representation.

H-LeWM:
Given current waypoint latent + macro-action,
predict a farther-away waypoint latent.
```

The hierarchy exists because flat planning over many primitive actions is hard. Prediction errors build up over long rollouts, and searching over many low-level actions becomes expensive.

So H-LeWM splits planning into two levels:

```text
High level:
Choose useful future subgoals using macro-actions.

Low level:
Use primitive actions to reach the next subgoal.
```

---

## Quick code index

Every step below has a **Code:** callout pointing to the exact symbol that implements it. This table is the same information at a glance:


| Step                | What it does                       | Code                                                                                                          |
| ------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Train Step 1        | Pick waypoint frames               | `[sample_waypoints](../waypoint_sampler.py:41)`                                                              |
| Train Step 2        | Encode waypoints with frozen E     | `[forward_high` (Step 2 block)](../hierarchical_lewm.py:387)                                                  |
| Train Step 3        | Encode action chunk → macro-action | `[ActionEncoder](../hierarchical_lewm.py:84)` · [call site](../hierarchical_lewm.py:393)                     |
| Train Step 4        | Predict next waypoint latent       | `[HighLevelPredictor](../hierarchical_lewm.py:185)` · [call site](../hierarchical_lewm.py:404)                |
| Train Step 5        | L1 loss + optimisation             | [loss](../hierarchical_lewm.py:413) · `[train_hierarchical_lewm](../hierarchical_lewm.py:502)`                |
| Whole train forward | All five train steps in one place  | `[HierarchicalLeWM.forward_high](../hierarchical_lewm.py:359)`                                                |
| Test Step 1         | Encode current + goal images       | `[HierarchicalPolicy._encode](../plan_hierarchical.py:86)`                                                    |
| Test Step 2         | Sample macro-action sequences      | `[plan` outer init](../hierarchical_plan.py:139)                                                              |
| Test Step 3         | Roll P² forward                    | `[_rollout_high](../hierarchical_lewm.py:429)` · [call site](../hierarchical_plan.py:149)                     |
| Test Step 4         | Score outer plans by L1 to z_goal  | [outer_cost](../hierarchical_plan.py:151)                                                                     |
| Test Step 5         | Take first waypoint as subgoal     | [subgoal extraction](../hierarchical_plan.py:158)                                                             |
| Test Step 6         | Sample primitive-action sequences  | `[plan` inner init](../hierarchical_plan.py:167)                                                              |
| Test Step 7         | Roll P¹ forward                    | `[_rollout_low](../hierarchical_lewm.py:456)` · [call site](../hierarchical_plan.py:179)                      |
| Test Step 8         | Score inner plans by L1 to subgoal | [inner_cost](../hierarchical_plan.py:181)                                                                     |
| Test Step 9         | Execute first primitive action     | `[plan` return](../hierarchical_plan.py:187) · `[HierarchicalPolicy.get_action](../plan_hierarchical.py:102)` |
| CEM utility         | Diagonal-Gaussian CEM              | `[cem](../hierarchical_plan.py:60)`                                                                           |
| Stage-2 entry       | Hydra training driver              | `[train_hierarchical.py](../train_hierarchical.py)`                                                           |
| Eval entry          | Hydra evaluation driver            | `[plan_hierarchical.py](../plan_hierarchical.py)`                                                             |


---

## Components

### Frozen components from LeWM

These are already trained during the original LeWM stage:

```text
E:
Image encoder.
Converts an image into a latent vector.

P1:
Low-level latent predictor.
Predicts short-horizon next latents from primitive actions.
```

During hierarchical training, these are frozen:

```text
Do not update E.
Do not update P1.
```

> **Code:** the frozen LeWM lives inside `HierarchicalLeWM.jepa` ([hierarchical_lewm.py:288](../hierarchical_lewm.py:262)). The `train()` method is overridden so that even when the outer module is in training mode, the inner `jepa` stays in `.eval()` — preventing BatchNorm running stats from drifting.

### New hierarchical components

These are trained in the hierarchical stage:

```text
A_psi:
Macro-action encoder.
Compresses a chunk of primitive actions into one macro-action vector.

P2:
High-level waypoint predictor.
Predicts future waypoint latents using macro-actions.
```

The macro-action is usually a continuous vector, not a hand-written action label.

Example:

```text
l_k = [0.2, -1.1, 0.7, ...]
```

In the slide, this macro-action vector is 8-dimensional.

> **Code:** `[ActionEncoder](../hierarchical_lewm.py:84)` (A_psi) and `[HighLevelPredictor](../hierarchical_lewm.py:185)` (P²). Both are stored on `HierarchicalLeWM` as `action_encoder_high` and `high_predictor`.

---

# Train methodology

## Training data

Start with offline trajectories:

```text
image, action, image, action, image, action, ...
```

More explicitly:

```text
o_t, a_t, o_{t+1}, a_{t+1}, o_{t+2}, ...
```

where:

```text
o_t = image at time t
a_t = primitive action at time t
```

---

## Step 1: Choose waypoint pairs

> **Code:** `[sample_waypoints](../waypoint_sampler.py:41)` in `hierarchical_lewm.py`. Called once per batch from inside `[train_hierarchical_lewm](../hierarchical_lewm.py:502)`.

Instead of using every next frame, H-LeWM uses farther-apart waypoint frames.

```text
Given a current waypoint image o_k,
choose a future waypoint image o_{k+1}.
```

These waypoints are separated by a chunk of primitive actions:

```text
a_k, a_{k+1}, ..., a_{k+L}
```

So the training example looks like:

```text
current waypoint image
+ action chunk
+ future waypoint image
```

---

## Step 2: Encode waypoint images

> **Code:** the no-grad encode block inside `[forward_high](../hierarchical_lewm.py:387)` (`emb = self.jepa.encode(...)["emb"]`, then `waypoint_latents = emb[:, waypoint_idx]`).

Use the frozen LeWM encoder:

```text
Given current waypoint image o_k,
encode it into waypoint latent z_k.
```

```text
Given future waypoint image o_{k+1},
encode it into target waypoint latent z_{k+1}.
```

So:

```text
E(o_k)       -> z_k
E(o_{k+1})   -> z_{k+1}
```

Important: the encoder is frozen, so its latent space stays the same as LeWM's latent space.

---

## Step 3: Encode the primitive action chunk

> **Code:** the class `[ActionEncoder](../hierarchical_lewm.py:84)` is `A_psi`; it is called from the per-segment loop in `[forward_high](../hierarchical_lewm.py:393)`.

The macro-action encoder compresses many primitive actions into one vector.

```text
Given a chunk of primitive actions,
predict / output one macro-action vector.
```

So:

```text
A_psi(a_k, a_{k+1}, ..., a_{k+L}) -> l_k
```

where:

```text
l_k = latent macro-action
```

Simple intuition:

```text
The macro-action vector summarizes what the whole action chunk does.
```

It is not necessarily human-readable. It may not mean exactly "move left" or "pick up object." It is just a useful latent vector for predicting waypoint-level transitions.

---

## Step 4: Predict the next waypoint latent

> **Code:** the class `[HighLevelPredictor](../hierarchical_lewm.py:185)` is `P²`; it is called once on the whole sequence (teacher-forced, causal) inside `[forward_high](../hierarchical_lewm.py:404)`.

Use the high-level predictor:

```text
Given current waypoint latent z_k + macro-action l_k,
predict the next waypoint latent z_hat_{k+1}.
```

So:

```text
P2(z_k, l_k) -> z_hat_{k+1}
```

This is the main high-level prediction task.

---

## Step 5: Train with waypoint prediction loss

> **Code:** L1 loss line in `[forward_high](../hierarchical_lewm.py:413)`; training loop in `[train_hierarchical_lewm](../hierarchical_lewm.py:502)`.

Compare the predicted waypoint latent to the true future waypoint latent:

```text
Prediction target:
z_{k+1}

Model prediction:
z_hat_{k+1}
```

Train the new modules so that:

```text
z_hat_{k+1} is close to z_{k+1}
```

The train-time objective is:

```text
Given current waypoint image + a chunk of primitive actions,
predict the future waypoint latent.
```

Only these modules are updated:

```text
Train A_psi.
Train P2.
```

These remain frozen:

```text
Freeze E.
Freeze P1.
```

---

## Train-time summary

> **Code:** all five steps live together in `[HierarchicalLeWM.forward_high](../hierarchical_lewm.py:359)` — read top-to-bottom, that single method implements the whole train pass.

```text
Input:
current waypoint image o_k
primitive action chunk a_k ... a_{k+L}
future waypoint image o_{k+1}

Frozen encoding:
E(o_k)       -> z_k
E(o_{k+1})   -> z_{k+1}

Macro-action encoding:
A_psi(a_k ... a_{k+L}) -> l_k

High-level prediction:
P2(z_k, l_k) -> z_hat_{k+1}

Loss:
make z_hat_{k+1} close to z_{k+1}

Updated:
A_psi and P2

Frozen:
E and P1
```

Core training sentence:

```text
Given current waypoint latent + encoded action chunk,
predict the next waypoint latent.
```

---

# Test methodology

At test time, the model receives:

```text
current image
goal image
```

It must choose real primitive actions to move from the current image toward the goal image.

The key difference from LeWM is that H-LeWM uses two planners:

```text
Outer planner:
Plans with macro-actions and predicts subgoals.

Inner planner:
Plans with primitive actions to reach the chosen subgoal.
```

> **Code:** the whole two-level planner is `[plan](../hierarchical_plan.py:100)` in `hierarchical_plan.py`. The MPC replan loop around it lives in `[HierarchicalPolicy.get_action](../plan_hierarchical.py:102)`.

---

## Step 1: Encode current image and goal image

> **Code:** `[HierarchicalPolicy._encode](../plan_hierarchical.py:86)` (called twice — once for the current image, once for the goal image — inside `[get_action](../plan_hierarchical.py:102)`).

Use the frozen encoder:

```text
Given current image,
encode it into z_now.
```

```text
Given goal image,
encode it into z_goal.
```

So:

```text
E(current image) -> z_now
E(goal image)    -> z_goal
```

---

# Outer planner: high-level planning

The outer planner searches over macro-actions.

> **Code:** the entire outer planner is the first half of `[plan](../hierarchical_plan.py:136)`. It uses the generic `[cem](../hierarchical_plan.py:60)` utility plus a custom `[outer_cost](../hierarchical_plan.py:144)` closure.

## Step 2: Sample macro-action sequences

> **Code:** initial Gaussian set up at [hierarchical_plan.py:139](../hierarchical_plan.py:139); sampling itself happens inside `[cem](../hierarchical_plan.py:60)`.

At test time, we do not know the true primitive action chunk yet.

So we sample macro-action vectors directly.

Example:

```text
l_1 = [0.3, -0.8, 1.1, ...]
l_2 = [-0.1, 0.4, 0.6, ...]
l_3 = [1.0, -0.2, -0.5, ...]
```

A candidate high-level plan is a sequence of macro-actions:

```text
Plan A: l_1, l_2, l_3
Plan B: l_1, l_2, l_3
Plan C: l_1, l_2, l_3
...
```

These are sampled by CEM.

Simple meaning:

```text
Try many possible abstract future plans.
```

---

## Step 3: Roll P2 forward

> **Code:** `[HierarchicalLeWM._rollout_high](../hierarchical_lewm.py:429)` does the actual autoregressive rollout; called from inside `[outer_cost](../hierarchical_plan.py:149)`.

For each sampled macro-action sequence, use P2 like a high-level simulator.

```text
Start at z_now.
Apply macro-action l_1 through P2 -> predict z_hat_1.
Apply macro-action l_2 through P2 -> predict z_hat_2.
Apply macro-action l_3 through P2 -> predict z_hat_3.
```

So:

```text
z_now  + l_1 -> z_hat_1
z_hat_1 + l_2 -> z_hat_2
z_hat_2 + l_3 -> z_hat_3
```

This is called rolling the model forward.

Simple meaning:

```text
Imagine where each abstract macro-action plan would take us.
```

---

## Step 4: Score each high-level plan

> **Code:** the L1 distance line inside `[outer_cost](../hierarchical_plan.py:151)`.

Compare the final predicted waypoint latent to the goal latent:

```text
distance(z_hat_3, z_goal)
```

A good high-level plan is one whose final predicted latent is close to the goal latent.

```text
Best macro-action plan =
the one whose predicted ending is closest to z_goal.
```

---

## Step 5: Choose the first predicted waypoint as the subgoal

> **Code:** subgoal extraction at [hierarchical_plan.py:158](../hierarchical_plan.py:158): `z_sg = model._rollout_high(z_init, best_mac.unsqueeze(0))[:, 0].squeeze(0)`.

The model does not directly execute the macro-action.

Instead, it takes the first predicted high-level waypoint:

```text
z_hat_1
```

and uses it as the next subgoal.

Simple meaning:

```text
The high-level planner says:
"Do not try to reach the final goal immediately.
First, try to reach this useful intermediate latent state."
```

Core outer-planner sentence:

```text
Given current latent + goal latent,
search over macro-actions to choose a useful subgoal.
```

---

# Inner planner: low-level planning

The inner planner uses the original frozen LeWM predictor P1.

> **Code:** the entire inner planner is the second half of `[plan](../hierarchical_plan.py:164)`. It uses `[cem](../hierarchical_plan.py:60)` again, this time with `[inner_cost](../hierarchical_plan.py:172)`.

## Step 6: Sample primitive action sequences

> **Code:** initial primitive-action Gaussian at [hierarchical_plan.py:167](../hierarchical_plan.py:167); sampling itself happens inside `[cem](../hierarchical_plan.py:60)`.

Now the goal is not the final goal image. The goal is the subgoal chosen by the outer planner:

```text
subgoal = z_hat_1
```

The inner planner samples many primitive action sequences:

```text
Plan A: move forward, move forward, close gripper
Plan B: move left, move forward, close gripper
Plan C: move right, move forward, open gripper
...
```

Simple meaning:

```text
Try many possible real action plans in imagination.
```

---

## Step 7: Roll P1 forward

> **Code:** `[HierarchicalLeWM._rollout_low](../hierarchical_lewm.py:456)` does the actual autoregressive rollout; called from inside `[inner_cost](../hierarchical_plan.py:179)`.

For each primitive action sequence, use the frozen low-level predictor P1 like a simulator.

```text
Start at current latent z_now.
Apply first primitive action  -> predict z_hat_{t+1}.
Apply second primitive action -> predict z_hat_{t+2}.
Apply third primitive action  -> predict z_hat_{t+3}.
```

So:

```text
current latent + action 1 -> predicted next latent
predicted latent + action 2 -> predicted next latent
predicted latent + action 3 -> predicted next latent
```

Simple meaning:

```text
Imagine where each real action sequence would take us.
```

---

## Step 8: Score each primitive action plan

> **Code:** the L1 distance line inside `[inner_cost](../hierarchical_plan.py:181)`.

Compare the final predicted low-level latent to the subgoal:

```text
distance(predicted final latent, z_hat_1)
```

A good primitive action plan is one that gets close to the subgoal.

```text
Best primitive action plan =
the one whose predicted ending is closest to z_hat_1.
```

---

## Step 9: Execute only the first primitive action

> **Code:** `[plan` return statement](../hierarchical_plan.py:187) (`return best_act[0]`); the MPC replan layer that calls `plan()` again on the next tick is `[HierarchicalPolicy.get_action](../plan_hierarchical.py:102)`.

Even if the inner planner finds a whole sequence, the agent only executes the first action:

```text
execute first action from best primitive plan
```

Then the agent observes the new image and replans again from scratch.

Simple meaning:

```text
Plan a few steps ahead,
take one real step,
look again,
then plan again.
```

This is MPC-style replanning.

---

## Test-time summary

> **Code:** read `[plan](../hierarchical_plan.py:100)` top-to-bottom for steps 2-9, and `[HierarchicalPolicy.get_action](../plan_hierarchical.py:102)` for step 1 + the MPC loop.

```text
Input:
current image
goal image

Encode:
E(current image) -> z_now
E(goal image)    -> z_goal

Outer planner:
sample macro-action sequences
roll P2 forward
choose the sequence that ends closest to z_goal
take first predicted waypoint z_hat_1 as subgoal

Inner planner:
sample primitive action sequences
roll P1 forward
choose the sequence that ends closest to z_hat_1
execute the first primitive action

Repeat:
observe new image
replan
```

Core test-time sentence:

```text
Given current image + goal image,
first search over macro-actions to choose a subgoal,
then search over primitive actions to reach that subgoal.
```

---

# Why the hierarchy helps

## 1. Shorter low-level rollouts

Flat LeWM has to plan over many primitive actions directly.

```text
Long primitive rollout -> more prediction error
```

H-LeWM only asks the low-level model to reach the next subgoal.

```text
Short primitive rollout -> less prediction error
```

---

## 2. Smaller high-level search space

Instead of searching over many primitive actions, the high-level planner searches over a few low-dimensional macro-actions.

Example from the slide:

```text
3 macro-actions x 8 dimensions = 24 dimensions
```

That is easier than searching over a long sequence of primitive robot actions.

---

## 3. Better long-horizon planning

The high-level model handles coarse long-horizon structure:

```text
Where should I go next?
```

The low-level model handles detailed action execution:

```text
What primitive actions get me there?
```

Together:

```text
H-LeWM plans far using abstract macro-actions
and acts precisely using primitive actions.
```

---

# Final compact version

```text
H-LeWM train:
Given current waypoint latent + encoded action chunk,
predict the next waypoint latent.

H-LeWM test:
Given current image + goal image,
use outer CEM over macro-actions to pick a subgoal,
then use inner CEM over primitive actions to reach that subgoal.
```

> **Code map:** train flow → `[HierarchicalLeWM.forward_high](../hierarchical_lewm.py:359)`; test flow → `[plan](../hierarchical_plan.py:100)`; top-level drivers → `[train_hierarchical.py](../train_hierarchical.py)` and `[plan_hierarchical.py](../plan_hierarchical.py)`.

