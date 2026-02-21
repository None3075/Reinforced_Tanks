from reward import reward_function0
from stable_baselines3.common.utils import FloatSchedule

# WanDB integration
USE_WANDB = False
PROJECT_NAME = "Project_name"
ENTITY_NAME = "Entity_name"

# Process Mode ["train", "render", "debug", "test", "choice"]
PROCESS_MODE = "choice"

# Test Mode Settings set Blue_Model to None to test agains all the models in enemy_models folder
TRIALS = 5
RED_MODEL = "models/model_1"
BLUE_MODEL = None #"models/model_2"
RENDER_TESTING = False

# Number of instances
N_INSTANCES = 1

# Render training process this can only be true if N_INSTANCES == 1
RENDER_TRAINING = True

# Random start
RANDOM_START = True

# Enemy deterministic behavior (IMPORTANT: set this to True for testing)
ENEMY_DETERMINISTIC = False

# Number of steps per epoch
TIMESTEPS: int = 1e6

# Number of epoch per learning step
EPOCHS = [5]

# Reward Functions of each learning step
REWARD_FUNCTIONS = [reward_function0] 

# Number of episodes between checkpoints
CHECKPOINT_STRIDE = 1

# Number of episodes between visualizations
VISUALIZATION_STRIDE = 1

# The number of steps maximun for a match if the game is not ended by kill the whole enemy team
TIME_LIMIT = 4000

# Arguments for the PPO model check https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html other hyperparameters can not be changed
learning_rate: float = FloatSchedule(0.0025)
n_steps: int =  2000
batch_size: int = 200
n_epochs: int = 22
gamma: float = 0.99
gae_lambda: float = 0.95
clip_range: float = FloatSchedule(0.2) 
clip_range_vf: float = FloatSchedule(0.2) 
normalize_advantage: bool = True
ent_coef: float = 0.02 
vf_coef: float = 0.5  
max_grad_norm: float = 0.5
use_sde: bool = False
sde_sample_freq: int = -1
stats_window_size: int = 100
verbose: int = 1
seed: int = None
target_kl: float = None#0.5

# Args to be passed to the whole program
args = {
    "use_wandb": USE_WANDB,
    "project_name": PROJECT_NAME,
    "entity_name": ENTITY_NAME,
    "test_trials": TRIALS,
    "red_model": RED_MODEL,
    "blue_model": BLUE_MODEL,
    "process_mode": PROCESS_MODE,
    "enemy_deterministic": ENEMY_DETERMINISTIC,
    "n_instances": N_INSTANCES,
    "render_training": RENDER_TRAINING,
    "render_testing": RENDER_TESTING,
    "timesteps": TIMESTEPS * 2,
    "epochs": EPOCHS,
    "reward_functions": REWARD_FUNCTIONS,
    "checkpoint_stride": CHECKPOINT_STRIDE,
    "visualization_stride": VISUALIZATION_STRIDE,
    "time_limit": TIME_LIMIT * 2,
    "random_start": RANDOM_START,
    "learning_rate": learning_rate,
    "n_steps": n_steps * 2,
    "batch_size": batch_size,
    "n_epochs": n_epochs,
    "gamma": gamma,
    "gae_lambda": gae_lambda,
    "clip_range": clip_range,
    "clip_range_vf": clip_range_vf,
    "normalize_advantage": normalize_advantage,
    "ent_coef": ent_coef,
    "vf_coef": vf_coef,
    "max_grad_norm": max_grad_norm,
    "use_sde": use_sde,
    "sde_sample_freq": sde_sample_freq,
    "stats_window_size": stats_window_size,
    "verbose": verbose,
    "seed": seed,
    "target_kl": target_kl,
}
