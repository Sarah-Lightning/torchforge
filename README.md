# TorchForge RL Training Template

Train policies with **PPO** or **GRPO** using [TorchForge](https://github.com/meta-pytorch/torchforge)'s ActorMesh for distributed rollout generation, reward scoring, and policy updates — orchestrated as long-running actors instead of one-shot batch jobs.

Built for RL loops where generation and training need to overlap continuously across GPUs, rather than run as discrete sweep stages. A small reference environment + reward function is included so the loop runs end-to-end on day one — swap in your own environment, policy, and reward model when ready.

## How it works

TorchForge's `ActorMesh` spins up a set of long-lived actors across your GPU pool:

- **Rollout actors** — generate trajectories from the current policy weights
- **Reward actor** — scores trajectories (built-in toy reward fn, or your own model/service)
- **Trainer actor** — computes the PPO/GRPO loss, takes the optimizer step, and broadcasts updated weights

Unlike a batch job, these actors stay alive for the whole run. Rollout actors keep sampling against the last-synced policy while the trainer computes the next update, so GPUs doing inference aren't idle while GPUs doing gradient updates run, and vice versa. Lightning manages the mesh lifecycle (provisioning, fault recovery, teardown); you own the RL loop logic in `train.py` and the actor definitions in `actors/`.

## Key features

- **ActorMesh topology, not a job queue** — rollout, reward, and trainer actors run concurrently; weight syncs and trajectory hand-offs happen over RPC, not via disk between stages.
- **Async generation/training overlap** — no strict generate-then-train phases.
- **PPO and GRPO out of the box** — switch the advantage estimator/loss in `config.yaml`; both share the same actor topology.
- **Fault-tolerant by default** — a dropped rollout actor restarts and rejoins the mesh without restarting the trainer or losing the current checkpoint.
- **Elastic rollout scaling** — `num_rollout_actors` scales generation throughput independently of trainer GPU count.
- **Tracking built in** — reward curves, KL-to-reference, and per-actor throughput stream to one dashboard; checkpoints save on a fixed cadence as artifacts.

## Repo structure

```
.
├── config.yaml              # algo (ppo/grpo), mesh sizing, checkpoint paths, reward config
├── train.py                  # entrypoint: builds the mesh, runs the training loop
├── submit_job.py              # launches the run as a Lightning job on the configured mesh
├── env.py                     # environment/dataset interface (swap for your task)
├── actors/
│   ├── __init__.py
│   ├── rollout_actor.py        # generates trajectories from current policy weights
│   ├── reward_actor.py         # scores trajectories (toy reward fn or external reward model)
│   └── trainer_actor.py        # PPO/GRPO loss, optimizer step, weight broadcast
├── utils/
│   ├── __init__.py
│   ├── advantages.py           # GAE (PPO) and group-relative (GRPO) advantage estimators
│   └── logging_utils.py        # lightweight metrics logger
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure the run** — edit `config.yaml` with your policy checkpoint, reward source, and mesh sizing (rollout actor count, trainer GPU count).
3. **Point at your task** — edit `env.py` to wrap your environment/dataset. A toy reference task (`ToyPromptEnv`) is included so the loop runs immediately.
4. **Smoke test locally** (small mesh, few steps):
   ```bash
   python train.py --config config.yaml --steps 50 --num-rollout-actors 2 --dry-run
   ```
5. **Submit the full run** on Lightning:
   ```bash
   python submit_job.py --config config.yaml
   ```

## Mesh sizing guidance

| Algo | Rollout : Trainer GPU ratio | Notes |
|------|------------------------------|-------|
| PPO  | ~3:1                         | Rollout-bound when `max_new_tokens` is large; scale rollout actors first. |
| GRPO | ~4:1                         | Group sampling (`group_size`) multiplies rollout cost per prompt; size rollout pool accordingly. |

Start with the smoke-test ratio above, then watch the dashboard — if rollout actors are the bottleneck (trainer actor idle waiting on trajectories), add more rollout actors before adding trainer GPUs.

## Swapping in your own reward model

Set `reward.mode: external` in `config.yaml` and point `reward.external_endpoint` at an HTTP/RPC endpoint (e.g. a LitServe-hosted reward model). `actors/reward_actor.py` will call it instead of the built-in toy reward function — no changes needed elsewhere in the loop.

## License

Apache-2.0
