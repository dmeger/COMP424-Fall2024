from world import World, PLAYER_1_NAME, PLAYER_2_NAME
import argparse
from utils import all_logging_disabled
import logging
import numpy as np
import datetime
from multiprocessing import Pool, cpu_count

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_1", type=str, default="random_agent")
    parser.add_argument("--player_2", type=str, default="random_agent")
    parser.add_argument("--board_size", type=int, default=None)
    parser.add_argument(
        "--board_size_min",
        type=int,
        default=6,
        help="In autoplay mode, the minimum board size",
    )
    parser.add_argument(
        "--board_size_max",
        type=int,
        default=12,
        help="In autoplay mode, the maximum board size",
    )
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_delay", type=float, default=0.4)
    parser.add_argument("--display_save", action="store_true", default=False)
    parser.add_argument("--display_save_path", type=str, default="plots/")
    parser.add_argument("--autoplay", action="store_true", default=False)
    parser.add_argument("--autoplay_runs", type=int, default=100)
    parser.add_argument(
        "--num_cores",
        type=int,
        default=None,
        help="Number of CPU cores to use. Default is all available cores.",
    )
    args = parser.parse_args()
    return args


def run_single_game(args_tuple):
    """Helper function to run a single game for parallel processing"""
    # Disable all logging for the individual game execution
    with all_logging_disabled():
        game_id, args, valid_board_sizes = args_tuple

        # Create a new simulator instance for this game
        simulator = Simulator(args)

        # Determine if players should be swapped for this game
        swap_players = game_id % 2 == 0

        # Select random board size
        board_size = valid_board_sizes[np.random.randint(len(valid_board_sizes))]

        # Run the game
        p0_score, p1_score, p0_time, p1_time = simulator.run(
            swap_players=swap_players, board_size=board_size
        )

        # Adjust scores if players were swapped
        if swap_players:
            p0_score, p1_score, p0_time, p1_time = p1_score, p0_score, p1_time, p0_time

        # Return results
        return {
            "p0_score": p0_score,
            "p1_score": p1_score,
            "p0_time": p0_time,
            "p1_time": p1_time,
            "board_size": board_size,
            "was_swapped": swap_players,
        }


class Simulator:
    """
    Entry point of the game simulator.

    Parameters
    ----------
    args : argparse.Namespace
    """

    def __init__(self, args):
        self.args = args
        # Only play on even-sized boards
        self.valid_board_sizes = [
            i
            for i in range(self.args.board_size_min, self.args.board_size_max + 1)
            if i % 2 == 0
        ]

    def reset(self, swap_players=False, board_size=None):
        """
        Reset the game

        Parameters
        ----------
        swap_players : bool
            if True, swap the players
        board_size : int
            if not None, set the board size
        """
        if board_size is None:
            board_size = self.args.board_size
        if swap_players:
            player_1, player_2 = self.args.player_2, self.args.player_1
        else:
            player_1, player_2 = self.args.player_1, self.args.player_2

        self.world = World(
            player_1=player_1,
            player_2=player_2,
            board_size=board_size,
            display_ui=self.args.display,
            display_delay=self.args.display_delay,
            display_save=self.args.display_save,
            display_save_path=self.args.display_save_path,
            autoplay=self.args.autoplay,
        )

    def run(self, swap_players=False, board_size=None):
        """
        Run a single game until completion.

        Parameters
        ----------
        swap_players : bool
            if True, swap the players
        board_size : int
            if not None, set the board size

        Returns
        -------
        tuple
            (p0_score, p1_score, p0_time, p1_time)
        """
        self.reset(swap_players=swap_players, board_size=board_size)
        is_end, p0_score, p1_score = self.world.step()
        while not is_end:
            is_end, p0_score, p1_score = self.world.step()
        return p0_score, p1_score, self.world.p0_time, self.world.p1_time

    def autoplay(self):
        """
        Run multiple simulations of the gameplay in parallel and aggregate win %.
        Uses multiprocessing to utilize all available CPU cores for faster execution.
        """
        if self.args.display:
            logger.warning("Since running autoplay mode, display will be disabled")
        self.args.display = False

        # Determine number of cores to use
        num_cores = self.args.num_cores or cpu_count()
        num_cores = min(num_cores, self.args.autoplay_runs)

        logger.info(
            f"Running {self.args.autoplay_runs} games using {num_cores} cores..."
        )

        # Prepare arguments for parallel processing
        game_args = [(i, self.args, [6]) for i in range(self.args.autoplay_runs)]

        # Run games in parallel
        with Pool(num_cores) as pool:
            results = pool.map(run_single_game, game_args)

        # Aggregate results
        p1_win_count = 0
        p2_win_count = 0
        p1_times = []
        p2_times = []
        board_sizes_used = []

        for result in results:
            if result["p0_score"] > result["p1_score"]:
                p1_win_count += 1
            elif result["p0_score"] < result["p1_score"]:
                p2_win_count += 1
            else:  # Tie
                p1_win_count += 0.5
                p2_win_count += 0.5

            p1_times.extend(result["p0_time"])
            p2_times.extend(result["p1_time"])
            board_sizes_used.append(result["board_size"])

        # Calculate statistics
        p1_win_rate = p1_win_count / self.args.autoplay_runs
        p2_win_rate = p2_win_count / self.args.autoplay_runs
        p1_max_time = np.round(np.max(p1_times), 5)
        p2_max_time = np.round(np.max(p2_times), 5)

        # Report final results
        logger.info("\nResults Summary:")
        logger.info("=" * 50)
        logger.info(f"Total Games Played: {self.args.autoplay_runs}")
        logger.info(f"Board Sizes Used: {sorted(set(board_sizes_used))}")
        logger.info(f"\nPlayer 1 ({self.args.player_1}):")
        logger.info(f"  Win Rate: {p1_win_rate:.2%}")
        logger.info(f"  Maximum Turn Time: {p1_max_time} seconds")
        logger.info(f"\nPlayer 2 ({self.args.player_2}):")
        logger.info(f"  Win Rate: {p2_win_rate:.2%}")
        logger.info(f"  Maximum Turn Time: {p2_max_time} seconds")
        logger.info("=" * 50)

        # Create a temporary world just to get player names for the CSV
        with all_logging_disabled():
            self.reset(
                board_size=6
            )  # This creates self.world with correct player names

        fname = (
            "tournament_results/"
            + self.world.player_1_name
            + "_vs_"
            + self.world.player_2_name
            + "_at_"
            + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + ".csv"
        )
        with open(fname, "w") as fo:
            fo.write(
                "P1Name,P2Name,NumRuns,P1WinPercent,P2WinPercent,P1RunTime,P2RunTime\n"
            )
            fo.write(
                f"{self.world.player_1_name},{self.world.player_2_name},{self.args.autoplay_runs},{p1_win_count / self.args.autoplay_runs},{p2_win_count / self.args.autoplay_runs},{np.round(np.max(p1_times),5)},{np.round(np.max(p2_times),5)}\n"
            )


if __name__ == "__main__":
    args = get_args()
    simulator = Simulator(args)
    if args.autoplay:
        simulator.autoplay()
    else:
        simulator.run()
