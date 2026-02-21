import os
import pygame
import pygame_gui
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import json
import pygame
import shutil
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.input import keys, press_key, release_key, get_pressed
from src.ObservationParser import OBSParser
from src.GameRenderer import GameRenderer
from src.Trainer import Trainer
from src.ReinforcedTanksEnv import ReinforcedTanksEnv
from src.feature_extractor.OpmizedModel import OptimizedModel
from src.Game import Game
from src.Agent import Agent
from utils.Logger import DefaultLogger

pygame.init()


def make_env(time_limit, random_start):
    """
    Factory function to create environment instances.
    This function will be called in each subprocess.
    """
    def _init():
        
        env = ReinforcedTanksEnv(
            time_limit=time_limit, 
            random_start=random_start
        )
        return Monitor(env)
    return _init

class MenuWindow:
    width: int
    height: int
    title: str
    logger: DefaultLogger

    trainer: Trainer
    window: pygame.Surface
    clock: pygame.time.Clock
    manager: pygame_gui.UIManager

    args: dict

    def __init__(self, width: int, height: int, title: str, fps: int, args: dict, show_logs: bool = True):
        self.width = width
        self.height = height
        self.title = title
        self.logger = DefaultLogger(path="logs", name="MenuWindow", level=os.getenv("LOG_LEVEL", "DEBUG"), print_in_terminal=show_logs)
        self.fps = fps
        self.args = args

        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()

        policy_kwargs = dict(
            features_extractor_class=OptimizedModel,
            features_extractor_kwargs=dict(features_dim=64, 
                                            T_B_hidden_layers = (256, 128, 64),
                                            T_B_embedding_dim= 32,
                                            T_T_hidden_layers = (256, 128, 64),
                                            T_T_embedding_dim = 32,
                                           ),
            net_arch = dict(
                pi=[32, 16, 8],
                vf=[32, 16, 8]
            )

        )

        env = ReinforcedTanksEnv(time_limit=args["time_limit"], random_start=args["random_start"], rendering=args["render_training"])
        if args["n_instances"] > 1:
            env_fns = [make_env(args["time_limit"], args["random_start"]) for i in range(args["n_instances"])]
            model = Agent(env=SubprocVecEnv(env_fns), args=args,
                        policy_kwargs=policy_kwargs
                        )
        else:
            model = Agent(env=env, args=args,
                        policy_kwargs=policy_kwargs
                        )
        self.trainer = Trainer(model=model, env=env, args=args)
        
        # GUI
        self.manager = pygame_gui.UIManager((self.width, self.height))

        self.center_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (500, 450)),
                                                        anchors={"center": "center"},
                                                        manager=self.manager)
        
        self.title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, -150), (200, 50)),
                                                       text="Reinforced Tanks",
                                                       anchors={"center": "center"},
                                                       manager=self.manager,
                                                       container=self.center_panel)

        self.train_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((0, -90), (200, 50)),
                                             text='Train',
                                             anchors={"center": "center"},
                                             manager=self.manager,
                                             container=self.center_panel)
        
        self.test_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((0, -30), (200, 50)),
                                             text='Test',
                                             anchors={"center": "center"},
                                             manager=self.manager,
                                             container=self.center_panel)
        
        self.replay_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((0, 30), (200, 50)),
                                             text='Replay',
                                             anchors={"center": "center"},
                                             manager=self.manager,
                                             container=self.center_panel)
        
        self.versus_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((0, 90), (200, 50)),
                                             text='Versus',
                                             anchors={"center": "center"},
                                             manager=self.manager,
                                             container=self.center_panel)
        
        self.debug_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((0, 150), (100, 40)),
                                            text='Debug',
                                            anchors={"center": "center"},
                                            manager=self.manager,
                                            container=self.center_panel)
        
        self.exit_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((0, 195), (100, 40)),
                                             text='Exit',
                                             anchors={"center": "center"},
                                             manager=self.manager,
                                             container=self.center_panel)

    def run(self):

        while True:
            deltaTime : int = self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == self.exit_button:
                    if not self.args["render_training"]:
                        pygame.quit()
                    return
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.train_button:
                        self.train_mode()
                        return
                    if event.ui_element == self.test_button:
                        self.test_mode()
                        return
                    if event.ui_element == self.replay_button:
                        self.select_replay()
                        return
                    if event.ui_element == self.versus_button:
                        self.versus_mode()
                        return
                    if event.ui_element == self.debug_button:
                        self.debug_mode()
                        return

                self.manager.process_events(event)

            self.manager.update(deltaTime)
            self.window.fill((0, 0, 0))
            self.manager.draw_ui(self.window)
            pygame.display.flip()

    def train_mode(self):
        if not self.args["render_training"]:
            pygame.quit()
        epoch = 0
        for visualization_flag, save_flag, stage in self.trainer.train_model():
            
            epoch += 1
            
            if visualization_flag:
                total_reward = self.visualize_example_match(epoch)

            if save_flag:
                self.save_and_load_for_next_match(epoch, stage, total_reward)

        self.compress_replays()

    def test_mode(self) -> str:
        game = Game(time_limit=self.args["time_limit"])
        game.terrain.set_stage(2)
        if self.args["blue_model"] is not None:
            result_counter = self.test_models(
                game,
                red_model_path=self.args["red_model"],
                blue_model_path=self.args["blue_model"]
            )
        else:
            game.set_models(
                red_model=self.trainer.load_model(self.args["red_model"]),
                blue_model=None
            )
            model_files = [f for f in os.listdir("enemy_model/") if f.endswith('.zip')]
            result_counter = {"Red": 0, "Blue": 0, "Draw": 0}
            for enemy_model_file in model_files:
                enemy_model_path = os.path.join("enemy_model/", enemy_model_file)
                single_result_counter = self.test_models(
                    game,
                    red_model_path=self.args["red_model"],
                    blue_model_path=enemy_model_path
                )
                for key in result_counter:
                    result_counter[key] += single_result_counter[key]
        
        self.save_piechart(result_counter, out_path="test_results_distribution.png")
    
    def test_models(self, game: Game, red_model_path: str, blue_model_path: str) -> dict:
            result_counter = {"Red": 0, "Blue": 0, "Draw": 0}

            game.set_models(
                red_model=self.trainer.load_model(red_model_path) if "custom" not in red_model_path else self.args["custom_agent_red"],
                blue_model=self.trainer.load_model(blue_model_path) if "custom" not in blue_model_path else self.args["custom_agent_blue"]
            )

            for trial in tqdm(range(self.args["test_trials"]), desc=f"Testing {Path(red_model_path).stem} vs {Path(blue_model_path).stem}"):
                result = self.test_match(game)
                result_counter[result] += 1
            
            self.logger.info(f"Test results between Red: {red_model_path} and Blue: {blue_model_path} -> {result_counter}")

            return result_counter

    def save_piechart(self, result_counter: dict, out_path: str) -> None:
        labels = list(result_counter.keys())
        counts = list(result_counter.values())
        total = sum(counts)

        color_map = {
            "Red": "red",
            "Blue": "blue",
            "Draw": "gray",
        }
        colors = [color_map[label] for label in labels]

        labels_with_support = [
            f"{label} (n={count})"
            for label, count in zip(labels, counts)
        ]

        fig, ax = plt.subplots()

        ax.pie(
                counts,
                labels=labels_with_support,
                colors=colors,
                autopct="%1.1f%%"
            )
        ax.set_title(f"Results Distribution (total={total})")

        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

    def test_match(self, game: Game):
        renderer = GameRenderer() if self.args["render_testing"] else None
        done = False
        game.reset(random_start=True)
        obs = OBSParser.parse_obs(game.get_info())
        terrain_info = game.terrain.get_walls_info()
        step_counter = 0

        while not done:
            obs = OBSParser.get_obs(state=game.get_info())
            _ , done, time_counter = game.debug_step(
                red_action=game.red_player.take_action(obs) ,
                blue_action=game.blue_player.take_action(obs)
            )

            meta_info = {
                "Time": time_counter
            }

            for func in self.args["reward_functions"]:
                reward = func(obs, terrain_info, time_counter)
                meta_info[func.__name__] = reward

            if self.args["render_testing"]:
                renderer.render_observation(obs, terrain_info, meta_info)
            step_counter += 1
       
        return game.winner()

    def compress_replays(self):
        replays_folder = Path("replays")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outcome = replays_folder.parent / f"replays_{timestamp}"
        shutil.make_archive(
            base_name=str(outcome),
            format="zip",
            root_dir=str(replays_folder.parent),
            base_dir=str(replays_folder.name)
        )

    def save_and_load_for_next_match(self, epoch, stage, total_reward):
        saved_path = self.trainer.save_model(epoch=epoch, stage=stage,data={"total_reward": total_reward})
        enemies = os.path.join(os.getcwd(), "enemy_model/")
        enemy_models = [f for f in os.listdir(enemies) if f.endswith('.zip')]
        random_enemy_model = np.random.choice(enemy_models)
        enemy_model = self.trainer.load_model(os.path.join(enemies, random_enemy_model))
        self.trainer.env.game.blue_player.set_model(enemy_model)

    def visualize_example_match(self, epoch):
        renderer = GameRenderer() if not self.args["render_training"] else None
        obs_tuple = self.trainer.env.reset()
        obs = obs_tuple[0]  # Extract just the observation
        done = False
        total_reward = 0
        step_counter = 0
        self._init_replay_folder(epoch)
        walls_info = self.trainer.env.game.terrain.get_walls_info()
        while not done:
            action, _states = self.trainer.model.predict(obs, deterministic=True)
            obs_tuple = self.trainer.env.step(action)
            obs, reward, terminated, truncated, info = obs_tuple
            done = terminated or truncated
            total_reward += reward

            perspectived_obs = self.trainer.env._get_obs()
            if info["update_flag"]: 
                obs_filename = f"replays/{epoch}/obs_{step_counter}.json"
                if not self.args["render_training"]:
                    meta_info = {
                        "Step": step_counter,
                        "Total Reward": total_reward,
                    }
                    for func in self.args["reward_functions"]:
                        reward = func(self.trainer.env.game.red_player.sweet_swap(perspectived_obs,self.trainer.env.static_counter), walls_info, self.trainer.env.game.time_counter)
                        meta_info[func.__name__] = reward
                    renderer.render_observation(perspectived_obs, walls_info, meta_info)
                with open(obs_filename, "w") as f:
                    save = OBSParser.parse_obs(perspectived_obs)
                    json.dump(save, f)
                step_counter += 1
        if not self.args["render_training"]: renderer.close()
        return total_reward

    def _init_replay_folder(self, epoch):
        if os.path.exists(f"replays/{epoch}"):
            shutil.rmtree(f"replays/{epoch}")
        os.makedirs(f"replays/{epoch}", exist_ok=False)
                            
    def debug_mode(self):
        renderer = GameRenderer()
        env = Game(time_limit=12000)
        env.reset(random_start=True)
        terrain_info = env.terrain.get_walls_info()
        done = False
        
        selected_red_tank = 0
        selected_blue_tank = 0

        while not done:
            self.update_keys()
                
            selected_red_tank, selected_blue_tank = self._update_selected_tank(selected_red_tank=selected_red_tank, selected_blue_tank=selected_blue_tank)

            red_action, blue_action = self._debug_get_action(selected_red_tank=selected_red_tank, selected_blue_tank=selected_blue_tank)

            obs, done, time_counter = env.debug_step(red_action, blue_action)

            meta_info = {
                "R_T": selected_red_tank,
                "B_T": selected_blue_tank,
                "Time": time_counter
            }

            for func in self.args["reward_functions"]:
                reward = func(env.red_player.sweet_swap(obs, selected_red_tank), terrain_info, time_counter)
                meta_info[func.__name__] = reward

            renderer.render_observation(obs, env.terrain.get_walls_info(), meta_info)

    def update_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

            if event.type == pygame.KEYDOWN:
                    # opcional: ignorar autorepeat si lo tienes activado
                if not getattr(event, "repeat", False):
                    if event.key in keys:          # solo las teclas que te interesan
                        press_key(event.key)

            elif event.type == pygame.KEYUP:
                if event.key in keys:
                    release_key(event.key)

    def _update_selected_tank(self, selected_red_tank: int, selected_blue_tank: int) -> tuple[int, int]:

        if get_pressed(pygame.K_1) and selected_red_tank != 0:
            selected_red_tank = 0
        elif get_pressed(pygame.K_2) and selected_red_tank != 1:
            selected_red_tank = 1

        if get_pressed(pygame.K_3) and selected_blue_tank != 0:
            selected_blue_tank = 0
        elif get_pressed(pygame.K_4) and selected_blue_tank != 1:
            selected_blue_tank = 1
        return selected_red_tank, selected_blue_tank

    def _debug_get_action(self, selected_red_tank: int, selected_blue_tank: int)-> tuple[list[int], list[int]]:
        red_action = [0, 0, 0, 0, 0, 0]
        blue_action = [0, 0, 0, 0, 0, 0]

        # Handle red tank actions
        red_action = self.handle_red_input(selected_red_tank)
        blue_action = self.handle_blue_input(selected_blue_tank)

        return red_action, blue_action

    def handle_blue_input(self, selected_tank: int) -> list[int]:
        action = [0, 0, 0, 0, 0, 0]

        if get_pressed(pygame.K_UP):
            action[selected_tank * 3] = 1  # Move forward
        elif get_pressed(pygame.K_DOWN):
            action[selected_tank * 3] = 2  # Move backward
        else:
            action[selected_tank * 3] = 0  # Stay still

        if get_pressed(pygame.K_LEFT):
            action[selected_tank * 3 + 1] = 2  # Turn left
        elif get_pressed(pygame.K_RIGHT):
            action[selected_tank * 3 + 1] = 1  # Turn right
        else:
            action[selected_tank * 3 + 1] = 0  # No turn
        
        if get_pressed(pygame.K_RCTRL):
            action[selected_tank * 3 + 2] = 1  # Shoot
        else:
            action[selected_tank * 3 + 2] = 0  # Don't shoot

        return action
    
    def handle_red_input(self, selected_tank: int) -> list[int]:
        action = [0, 0, 0, 0, 0, 0]

        if get_pressed(pygame.K_w):
            action[selected_tank * 3] = 1  # Move forward
        elif get_pressed(pygame.K_s):
            action[selected_tank * 3] = 2  # Move backward
        else:
            action[selected_tank * 3] = 0  # Stay still

        if get_pressed(pygame.K_a):
            action[selected_tank * 3 + 1] = 2  # Turn left
        elif get_pressed(pygame.K_d):
            action[selected_tank * 3 + 1] = 1  # Turn right
        else:
            action[selected_tank * 3 + 1] = 0  # No turn
        
        if get_pressed(pygame.K_SPACE):
            action[selected_tank * 3 + 2] = 1  # Shoot
        else:
            action[selected_tank * 3 + 2] = 0  # Don't shoot

        return action

    def select_replay(self):
        
        games_folder = "replays"
        games = [f for f in os.listdir(games_folder) if os.path.isdir(os.path.join(games_folder, f))]
        games.sort()

        if not games:
            self.logger.info("No replays found in", games_folder)
            return

        # Clear current UI and show a replay selection panel
        self.manager.clear_and_reset()

        replay_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 0), (500, 420)),
            anchors={"center": "center"},
            manager=self.manager
        )

        selection_list = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((10, 10), (480, 320)),
            item_list=games,
            manager=self.manager,
            container=replay_panel
        )

        play_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((100, 340), (120, 50)),
            text='Play',
            manager=self.manager,
            container=replay_panel
        )

        cancel_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((280, 340), (120, 50)),
            text='Cancel',
            manager=self.manager,
            container=replay_panel
        )

        selected_game = None

        # Simple blocking UI loop until user picks Play or Cancel
        while True:
            deltaTime = self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == play_button:
                        # get the currently selected item from the selection list
                        selected_game = selection_list.get_single_selection()
                        if selected_game:
                            game_path = os.path.join(games_folder, selected_game)
                            self.logger.info(f"Selected game: {selected_game}")
                            # Call renderer and return
                            GameRenderer.renderGame(game_path)
                            return
                        else:
                            self.logger.info("No game selected.")
                    if event.ui_element == cancel_button:
                        # go back to menu
                        return

                if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION and event.ui_element == selection_list:
                    # update selection variable (optional)
                    selected_game = selection_list.get_single_selection()

                self.manager.process_events(event)

            self.manager.update(deltaTime)
            self.window.fill((0, 0, 0))
            self.manager.draw_ui(self.window)
            pygame.display.flip()
    
    def versus_mode(self):
        """Select two models and play them against each other with rendering."""
        # Get available models
        model_folder = "models/versus_models"
        if not os.path.exists(model_folder):
            self.logger.info("No checkpoints folder found")
            return
        
        # Collect all model files recursively
        model_files = []
        for root, dirs, files in os.walk(model_folder):
            for file in files:
                if file.endswith('.zip'):
                    model_files.append(os.path.join(root, file))
        
        if len(model_files) < 2:
            self.logger.info("Need at least 2 models for versus mode")
            return
        
        # Sort models
        model_files.sort()
        
        # Create selection UI
        self.manager.clear_and_reset()
        
        versus_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 0), (700, 500)),
            anchors={"center": "center"},
            manager=self.manager
        )
        
        title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (680, 30)),
            text='Select Red Team Model',
            manager=self.manager,
            container=versus_panel
        )
        
        red_selection = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((10, 50), (330, 300)),
            item_list=[Path(m).name for m in model_files],
            manager=self.manager,
            container=versus_panel
        )
        
        blue_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((360, 10), (330, 30)),
            text='Select Blue Team Model',
            manager=self.manager,
            container=versus_panel
        )
        
        blue_selection = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((360, 50), (330, 300)),
            item_list=[Path(m).name for m in model_files],
            manager=self.manager,
            container=versus_panel
        )
        
        play_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((200, 370), (120, 50)),
            text='Play',
            manager=self.manager,
            container=versus_panel
        )
        
        cancel_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((380, 370), (120, 50)),
            text='Cancel',
            manager=self.manager,
            container=versus_panel
        )
        
        # Selection loop
        while True:
            deltaTime = self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == play_button:
                        red_model_name = red_selection.get_single_selection()
                        blue_model_name = blue_selection.get_single_selection()
                        
                        if red_model_name and blue_model_name:
                            # Find full paths
                            red_path = next((m for m in model_files if Path(m).name == red_model_name), None)
                            blue_path = next((m for m in model_files if Path(m).name == blue_model_name), None)
                            
                            if red_path and blue_path:
                                self.logger.info(f"Starting versus: {red_model_name} vs {blue_model_name}")
                                self._play_versus_match(red_path, blue_path)
                            return
                        else:
                            self.logger.info("Please select both red and blue models")
                    
                    if event.ui_element == cancel_button:
                        return
                
                self.manager.process_events(event)
            
            self.manager.update(deltaTime)
            self.window.fill((0, 0, 0))
            self.manager.draw_ui(self.window)
            pygame.display.flip()
    
    def _play_versus_match(self, red_model_path: str, blue_model_path: str):
        """Play a single match between two models with rendering."""
        
        self.trainer.env.game.set_models(
            red_model=self.trainer.load_model(red_model_path),
            blue_model=self.trainer.load_model(blue_model_path)
        )
        
        def dummy_reward_function(obs, terrain_info, time_counter):
            return 0
        
        self.trainer.load_model(red_model_path)
        self.trainer.env.reward_function = dummy_reward_function
        
        # Show example match code but without saving replays
        renderer = GameRenderer() if not self.args["render_training"] else None
        obs_tuple = self.trainer.env.reset()
        obs = obs_tuple[0]  # Extract just the observation
        done = False
        step_counter = 0
        walls_info = self.trainer.env.game.terrain.get_walls_info()
        while not done:
            action, _states = self.trainer.model.predict(obs, deterministic=True)
            obs_tuple = self.trainer.env.step(action)
            obs, reward, terminated, truncated, info = obs_tuple
            done = terminated or truncated

            perspectived_obs = self.trainer.env._get_obs()
            if info["update_flag"]: 
                if not self.args["render_training"]:
                    meta_info = {
                        "Step": step_counter
                    }
                    renderer.render_observation(perspectived_obs, walls_info, meta_info)
                step_counter += 1
                
        winner = self.trainer.env.game.winner()
        self._show_match_result(winner, Path(red_model_path).stem, Path(blue_model_path).stem)
    
    def _show_match_result(self, winner: str, red_name: str, blue_name: str):
        """Show match result dialog."""
        self.manager.clear_and_reset()
        
        result_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 0), (400, 250)),
            anchors={"center": "center"},
            manager=self.manager
        )
        
        result_text = f"Winner: {winner}\n\nRed: {red_name}\nBlue: {blue_name}"
        
        result_label = pygame_gui.elements.UITextBox(
            html_text=result_text.replace("\n", "<br>"),
            relative_rect=pygame.Rect((20, 20), (360, 150)),
            manager=self.manager,
            container=result_panel
        )
        
        ok_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((140, 180), (120, 50)),
            text='OK',
            manager=self.manager,
            container=result_panel
        )
        
        while True:
            deltaTime = self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == ok_button:
                        return
                
                self.manager.process_events(event)
            
            self.manager.update(deltaTime)
            self.window.fill((0, 0, 0))
            self.manager.draw_ui(self.window)
            pygame.display.flip()
