# Reinforced Tanks

A reinforcement learning project that trains AI agents to play a tank combat game using PPO (Proximal Policy Optimization) from Stable Baselines3.

## Requirements

- Python 3.12.3
- (OPTIONAL) CUDA-compatible GPU (recommended for faster training)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lingfeng555/ReinforcedTanks.git
   cd ReinforcedTanks
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pretrained enemy models:**
   
   Download the pretrained models from [Google Drive](https://drive.google.com/file/d/1TWBoHahXmayEuRaMOtI6nUsLt6_eGBSn/view?usp=drive_link) and extract them into the `enemy_model/` directory:
   ```bash
   # After downloading the zip file
   unzip <downloaded_file>.zip -d enemy_model/
   ```

## Running the Project

To start the application, run:

```bash
python main.py
```

This will launch the menu window where you can start training or play the game.

### Parallel Training

For faster training, you can run multiple training instances in parallel using `paraller_main.py`:

```bash
python paraller_main.py --instances 4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--instances` | `1` | Number of parallel training processes to spawn |

This script launches multiple `main.py` processes simultaneously, which can significantly speed up data collection when using algorithms like PPO. Each instance runs independently and collects experience in parallel.

**Note:** Ensure your system has sufficient CPU/GPU resources to handle multiple instances. For optimal performance, set `--instances` to match your available CPU cores or GPU capacity.

### Migrating Enemy Model Pools

Use `migrate_pools.py` to copy trained models from one folder to the enemy model pool. This is useful for adding new opponents to the training pool after completing a training run:

```bash
python migrate_pools.py --source_folder other_folder --destination_folder enemy_model
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source_folder` | (required) | Path to folder containing trained model `.zip` files |
| `--destination_folder` | `enemy_model` | Target folder for the enemy model pool |

The script automatically renumbers the models to avoid conflicts with existing files in the destination folder. For example, if `enemy_model/` already contains `model_0.zip` through `model_4.zip`, new models will be numbered starting from `model_5.zip`.

## Hyperparameters Configuration

All hyperparameters are configured in the [hyperparameters.py](hyperparameters.py) file. Below is a description of each parameter:

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_WANDB` | `False` | Enable Weights & Biases logging |
| `PROJECT_NAME` | `"NombreEjemplo"` | W&B project name |
| `ENTITY_NAME` | `"TeamName"` | W&B team/entity name |
| `PROCESS_MODE` | `"choice"` | Mode: `"train"`, `"render"`, `"debug"`, `"test"`, or `"choice"` |
| `N_INSTANCES` | `1` | Number of parallel environment instances for training |
| `RENDER_TRAINING` | `True` | Enable visualization during training (requires `N_INSTANCES=1`) |
| `RANDOM_START` | `True` | Randomize initial positions of tanks |
| `ENEMY_DETERMINISTIC` | `False` | Use deterministic enemy behavior (set `True` for testing) |
| `TIMESTEPS` | `1e6` | Number of steps per training epoch |
| `EPOCHS` | `[5]` | List of epochs per learning stage |
| `REWARD_FUNCTIONS` | `[reward_function0]` | Reward function(s) to use for each stage |
| `CHECKPOINT_STRIDE` | `1` | Save checkpoint every N epochs |
| `VISUALIZATION_STRIDE` | `1` | Visualize every N episodes |
| `TIME_LIMIT` | `4000` | Maximum steps per match before timeout |

### Testing Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRIALS` | `5` | Number of test trials to run |
| `RED_MODEL` | `"models/model_1"` | Path to the red team model |
| `BLUE_MODEL` | `None` | Path to blue team model (`None` tests against all models in `enemy_model/`) |
| `RENDER_TESTING` | `False` | Enable visualization during testing |

### PPO Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0025 | Learning rate for the optimizer |
| `n_steps` | 2048 | Number of steps per environment per update |
| `batch_size` | 64 | Minibatch size for each gradient update |
| `n_epochs` | 20 | Number of epochs when optimizing the surrogate loss |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | Factor for Generalized Advantage Estimation |
| `clip_range` | 0.2 | Clipping parameter for PPO |
| `clip_range_vf` | 0.2 | Clipping parameter for value function |
| `normalize_advantage` | True | Whether to normalize the advantage |
| `ent_coef` | 0.02 | Entropy coefficient for exploration |
| `vf_coef` | 0.5 | Value function coefficient in the loss |
| `max_grad_norm` | 0.5 | Maximum gradient norm for clipping |
| `use_sde` | False | Use State Dependent Exploration |
| `sde_sample_freq` | -1 | Sample new noise matrix every n steps |
| `stats_window_size` | 100 | Window size for logging statistics |
| `verbose` | 1 | Verbosity level (0: none, 1: info, 2: debug) |
| `seed` | None | Random seed for reproducibility |
| `target_kl` | None | Target KL divergence for early stopping |

### Modifying Hyperparameters

1. Open [hyperparameters.py](hyperparameters.py) in your editor
2. Modify the desired parameters
3. Save the file and restart training

**Example: Increase learning rate and reduce batch size:**
```python
learning_rate: float = FloatSchedule(0.005)  # Increased from 0.0025
batch_size: int = 32                          # Reduced from 64
```

**Example: Enable rendering for debugging:**
```python
N_INSTANCES = 1           # Must be 1 for rendering
RENDER_TRAINING = True    # Enable visualization
```

## Custom Reward Functions

Reward functions are defined in [reward.py](reward.py). Each function must follow this signature:

```python
def my_reward_function(obs: dict, terrain: dict, time: int) -> float:
    # obs contains:
    #   - "own_tanks": list of own tank states
    #   - "enemy_tanks": list of enemy tank states
    #   - "own_bullets": list of own bullet states
    #   - "enemy_bullets": list of enemy bullet states
    # terrain: terrain information
    # time: current timestep
    return reward_value
```

### Creating a Custom Reward Function

1. Define your function in `reward.py`:
   ```python
   def my_custom_reward(obs: dict, terrain: dict, time: int) -> float:
       n_own_alive = sum(tank[0] for tank in obs["own_tanks"])
       n_enemy_alive = sum(tank[0] for tank in obs["enemy_tanks"])
       
       if n_enemy_alive == 0:
           return 10  # Win bonus
       elif n_own_alive == 0:
           return -10  # Loss penalty
       
       return (n_own_alive - n_enemy_alive) / 3
   ```

2. Import and add it to `REWARD_FUNCTIONS` in `hyperparameters.py`:
   ```python
   from reward import my_custom_reward
   REWARD_FUNCTIONS = [my_custom_reward]
   ```

### Available Helper Functions

The `reward.py` file includes helper functions you can use:

| Function | Description |
|----------|-------------|
| `R1(obs, k)` | Position-based reward comparing distance to center |
| `R2(obs, bullet, r)` | Bullet trajectory reward based on enemy proximity |
| `R3(obs, r)` | Defensive reward based on enemy bullet trajectories |
| `distance(p1, p2)` | Euclidean distance between two points |
| `distance_to_center(tank)` | Distance from tank to map center |

## Monitoring Training

Training metrics are logged to:
- **TensorBoard:** `tensorboard/` directory
- **Weights & Biases:** `wandb/` directory

To view TensorBoard logs:
```bash
tensorboard --logdir tensorboard/
```

## Project Structure

```
ReinforcedTanks/
├── main.py              # Entry point
├── hyperparameters.py   # Training configuration
├── reward.py            # Reward function definitions
├── requirements.txt     # Python dependencies
├── src/                 # Source code
│   ├── Agent.py         # RL agent implementation
│   ├── Trainer.py       # Training loop
│   ├── ReinforcedTanksEnv.py  # Gymnasium environment
│   ├── GameRenderer.py  # Game visualization
│   └── ...
├── checkpoints/         # Saved model checkpoints
├── tensorboard/         # Training logs
└── assets/              # Game assets
```

## Match Rules

A match ends when one of the following conditions is met:

| Condition | Winner |
|-----------|--------|
| All enemy tanks destroyed | Team that eliminated the enemy |
| Time limit reached | Team with more tanks alive |
| Time limit + equal tanks | Team with smallest mean distance to map center |

**Tournament time limit:** 12,000 steps (~3.3 minutes at 60 FPS)

## Tournament Submission

### What You Can Modify

You have full freedom to modify any part of the codebase to develop your strategy:

- **Game parameters:** Adjust time limits, reward functions, or training conditions
- **Training strategy:** For aggressive agents, reduce `TIME_LIMIT` to emphasize kill rewards over positional rewards
- **Collaboration:** Share models with teammates or trade with other teams
- **Hyperparameter tuning:** Implement custom learning rate schedules or other optimizations

> **Note:** As long as your model can be loaded and used using the provided codebase, any approach is valid.

### Submission Requirements

Send an email to **reinforcedtanks@gmail.com** with:

| Item | Description |
|------|-------------|
| Model file | Your trained model (`.zip` format) |
| Team logo | Image file for tournament display |
| Team name | Your team's name |
| Members | Names of all team members |

**Email subject:** `<team_name>_final_submission`

### Tournament Format

Once submitted, your model will compete in a **single-elimination bracket** with:

| Setting | Value |
|---------|-------|
| Mode | Test mode (deterministic) |
| Matches per round | 100 |
| Starting positions | Randomized |
| Bracket order | Randomly generated |

The team that wins the most matches in each round advances to the next stage.

**Results announcement:** June 10th at University of Deusto (in-person event). More details and prizes will be announced on the website.