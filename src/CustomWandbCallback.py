import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Dict, Any

class CustomWandbCallback(WandbCallback):
    """Extended WandB callback that tracks win/loss ratios across environments"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulated_results = {"red_wins": 0, "blue_wins": 0, "draws": 0, "total": 0}
        self.last_log_step = 0
        self.log_frequency = 10000
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if hasattr(self.training_env, 'env_method'):
            try:
                results_list = self.training_env.env_method('get_episode_results')
                
                for env_results in results_list:
                    for key in self.accumulated_results:
                        self.accumulated_results[key] += env_results[key]
                
                self.training_env.env_method('reset_episode_results')
                
            except Exception as e:
                pass
        else:
            try:
                env_results = self.training_env.get_episode_results()
                for key in self.accumulated_results:
                    self.accumulated_results[key] += env_results[key]
                self.training_env.reset_episode_results()
            except Exception:
                pass
        
        if self.accumulated_results["total"] > 0 and (self.n_calls - self.last_log_step) >= self.log_frequency:
            self._log_win_loss_ratios()
            self.last_log_step = self.n_calls
        
        return result
    
    def _log_win_loss_ratios(self):
        """Calculate and log win/loss ratios to WandB"""
        total = self.accumulated_results["total"]
        if total == 0:
            return
        
        red_wins = self.accumulated_results["red_wins"]
        blue_wins = self.accumulated_results["blue_wins"]
        draws = self.accumulated_results["draws"]
        
        wandb.log({
            "performance/red_win_rate": red_wins / total,
            "performance/blue_win_rate": blue_wins / total,
            "performance/draw_rate": draws / total,
            "performance/red_wins_total": red_wins,
            "performance/blue_wins_total": blue_wins,
            "performance/draws_total": draws,
            "performance/episodes_total": total,
        }, step=self.n_calls)