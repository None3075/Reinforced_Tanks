import warnings

from hyperparameters import args
from src.MenuWindow import MenuWindow

warnings.simplefilter('default', DeprecationWarning)

## CANNOT BE CHANGED
MAP_HEIGHT = 1080 / 2
MAP_WIDTH = 1920 / 2

def main():
    if args["render_training"]:
        args["n_instances"] = 1
    else:
        print("Rendering the training process is only available with a single instance.")
    match args["process_mode"]:
        case "train":
            MenuWindow(MAP_WIDTH, MAP_HEIGHT, "Reinforced Tanks", fps=60, args=args).train_mode()
        case "choice":
            MenuWindow(MAP_WIDTH, MAP_HEIGHT, "Reinforced Tanks", fps=60, args=args).run()
        case "render":
            MenuWindow(MAP_WIDTH, MAP_HEIGHT, "Reinforced Tanks", fps=60, args=args).select_replay()
        case "debug":
            MenuWindow(MAP_WIDTH, MAP_HEIGHT, "Reinforced Tanks", fps=60, args=args).debug_mode()
        case "test":
            MenuWindow(MAP_WIDTH, MAP_HEIGHT, "Reinforced Tanks", fps=60, args=args).test_mode()
        case "versus":
            MenuWindow(MAP_WIDTH, MAP_HEIGHT, "Reinforced Tanks", fps=60, args=args).versus_mode()
    

if __name__ == "__main__":
    main()
