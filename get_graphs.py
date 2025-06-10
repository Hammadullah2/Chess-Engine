import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from chess_engine import Chess
import chess.engine
import concurrent.futures
from tqdm import tqdm
import traceback

class ChessAlgorithmTester:
    def __init__(self, stockfish_path, num_games=10, max_moves=100):
        self.stockfish_path = stockfish_path
        self.num_games = num_games
        self.max_moves = max_moves
        self.results = {
            "alpha-beta": {},
            "evolutionary": {},
            "pso": {}
        }
        
        # Parameter combinations to test
        self.alpha_beta_params = [
            {"depth": 2},
            {"depth": 3},
            {"depth": 4}
        ]
        
        self.evolutionary_params = [
            {"population_size": 20, "generations": 5},
            {"population_size": 30, "generations": 10},
            {"population_size": 40, "generations": 15}
        ]
        
        self.pso_params = [
            {"num_particles": 20, "iterations": 5},
            {"num_particles": 30, "iterations": 10},
            {"num_particles": 40, "iterations": 15}
        ]

    def get_stockfish_evaluation(self, board, engine, depth=10):
        """Get evaluation from Stockfish at the given depth using existing engine"""
        try:
            analysis = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = analysis['score'].white()
            
            if score.is_mate():
                mate_in = score.mate()
                return 1000 if mate_in > 0 else -1000
            else:
                return score.score() / 100.0  # Convert centipawns to pawns
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0

    def run_game(self, algorithm, params, game_idx):
        """Run a single game between the algorithm and Stockfish"""
        board_ai = Chess()
        board_stockfish = chess.Board()
        
        evaluations = []  # Stockfish evaluations after each move
        move_times = []   # Time taken for each AI move
        outcomes = []     # Game outcome at each move
        ai_moves = []     # AI moves made
        
        # Use a single engine instance per game
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as e:
            print(f"Failed to start Stockfish: {e}")
            return {
                "evaluations": [],
                "move_times": [],
                "outcomes": [],
                "result": 0,  # Consider engine failure as loss
                "moves": []
            }
        
        # Initial evaluation
        try:
            initial_eval = self.get_stockfish_evaluation(board_stockfish, engine)
            evaluations.append(initial_eval)
            outcomes.append(0)  # 0 = ongoing
        except Exception as e:
            print(f"Initial evaluation failed: {e}")
            evaluations.append(0.0)
            outcomes.append(0)
        
        move_count = 0
        game_result = 0  # Default to loss (consider no move as loss)

        try:
            while not board_stockfish.is_game_over(claim_draw=True) and move_count < self.max_moves:
                move_count += 1
                
                if board_ai.p_move == 1:  # AI's turn (white)
                    start_time = time.time()
                    move_ai = None
                    
                    try:
                        if algorithm == "alpha-beta":
                            move_ai = board_ai.get_alpha_beta_move(depth=params['depth'])
                        elif algorithm == "evolutionary":
                            move_ai = board_ai.evolutionary_algorithm(
                                population_size=params['population_size'],
                                generations=params['generations']
                            )
                        elif algorithm == "pso":
                            move_ai = board_ai.particle_swarm_optimization(
                                num_particles=params['num_particles'],
                                iterations=params['iterations']
                            )
                    except Exception as e:
                        print(f"Move generation failed: {e}")
                        # Consider move generation failure as loss
                        game_result = 0
                        break
                    
                    if move_ai is None:
                        print("AI returned no move - registering game as loss.")
                        game_result = 0
                        break
                    
                    time_taken = time.time() - start_time
                    move_times.append(time_taken)
                    
                    # Convert to UCI move string
                    from_sq = move_ai[0].lower()
                    to_sq = move_ai[1].lower()
                    promotion = move_ai[2] if move_ai[2] else ''
                    uci_move = from_sq + to_sq + promotion
                    ai_moves.append(uci_move)
                    
                    # Make the move on both boards
                    board_ai.move(move_ai[0], move_ai[1], move_ai[2])
                    chess_move = chess.Move.from_uci(uci_move)
                    board_stockfish.push(chess_move)
                    
                else:  # Stockfish's turn (black)
                    try:
                        result = engine.play(
                            board_stockfish, 
                            chess.engine.Limit(time=0.1),  # Reduced time for faster testing
                            info=chess.engine.INFO_SCORE
                        )
                        chess_move = result.move
                        board_stockfish.push(chess_move)
                        
                        # Convert to our move format
                        uci_str = chess_move.uci()
                        from_sq = uci_str[0:2]
                        to_sq = uci_str[2:4]
                        promotion = uci_str[4] if len(uci_str) > 4 else None
                        
                        # Make the move on our board
                        board_ai.move(from_sq, to_sq, promotion)
                    except Exception as e:
                        print(f"Stockfish move failed: {e}")
                        # Consider engine failure as win for AI
                        game_result = 1
                        break
                
                # Record evaluation after this move
                try:
                    current_eval = self.get_stockfish_evaluation(board_stockfish, engine)
                    evaluations.append(current_eval)
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    evaluations.append(0.0)
                
                # Track game outcome
                if board_stockfish.is_checkmate():
                    outcomes.append(1 if board_stockfish.turn == chess.BLACK else -1)
                    game_result = 1 if board_stockfish.turn == chess.BLACK else 0
                    break
                elif board_stockfish.is_stalemate() or board_stockfish.is_insufficient_material() or board_stockfish.is_fifty_moves() or board_stockfish.is_repetition():
                    outcomes.append(0)
                    game_result = 0.5
                    break
                else:
                    outcomes.append(0)  # Ongoing
            
            # Final game outcome if not already set
            if board_stockfish.is_checkmate():
                game_result = 1 if board_stockfish.turn == chess.BLACK else 0
            elif board_stockfish.is_variant_draw():
                game_result = 0.5
            else:  # Game ended by move limit
                game_result = 0.5
                
        except Exception as e:
            print(f"Error in game {game_idx}: {e}")
            traceback.print_exc()
            # Consider any exception as loss
            game_result = 0
        finally:
            try:
                engine.quit()
            except:
                pass
        
        return {
            "evaluations": evaluations,
            "move_times": move_times,
            "outcomes": outcomes,
            "result": game_result,
            "moves": ai_moves
        }

    def test_algorithm(self, algorithm):
        """Test a specific algorithm with all parameter combinations"""
        param_sets = []
        param_names = []
        
        if algorithm == "alpha-beta":
            param_sets = self.alpha_beta_params
            param_names = [f"Depth={p['depth']}" for p in param_sets]
        elif algorithm == "evolutionary":
            param_sets = self.evolutionary_params
            param_names = [f"Pop={p['population_size']}, Gen={p['generations']}" for p in param_sets]
        elif algorithm == "pso":
            param_sets = self.pso_params
            param_names = [f"Particles={p['num_particles']}, Iter={p['iterations']}" for p in param_sets]
        
        for param_idx, params in enumerate(param_sets):
            print(f"\nTesting {algorithm} with parameters: {params}")
            param_name = param_names[param_idx]
            
            # Initialize storage for this parameter set
            self.results[algorithm][param_name] = {
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "evaluation_data": [],
                "time_data": [],
                "moves": []
            }
            
            # Run games sequentially (parallel execution was causing issues)
            for i in tqdm(range(self.num_games), desc="Games"):
                game_result = self.run_game(algorithm, params, i)
                
                # Record outcome
                if game_result["result"] == 1:
                    self.results[algorithm][param_name]["wins"] += 1
                elif game_result["result"] == 0.5:
                    self.results[algorithm][param_name]["draws"] += 1
                else:
                    self.results[algorithm][param_name]["losses"] += 1
                
                # Store evaluation data
                self.results[algorithm][param_name]["evaluation_data"].append(
                    game_result["evaluations"]
                )
                
                # Store time data
                if game_result["move_times"]:
                    self.results[algorithm][param_name]["time_data"].append(
                        game_result["move_times"]
                    )
                
                # Store moves
                self.results[algorithm][param_name]["moves"].append(
                    game_result["moves"]
                )

    def run_all_tests(self):
        """Run tests for all algorithms"""
        print("Starting performance analysis...")
        
        # Test each algorithm
        for algorithm in ["alpha-beta", "evolutionary", "pso"]:
            print(f"\n{'='*50}")
            print(f"Testing {algorithm} algorithm")
            print(f"{'='*50}")
            self.test_algorithm(algorithm)
        
        print("\nAll tests completed!")

    def plot_performance(self):
        """Create various performance graphs"""
        # Create output directory
        os.makedirs("performance_graphs", exist_ok=True)
        
        # 1. Win/Draw/Loss Rates
        self.plot_outcome_rates()
        
        # 2. Evaluation Progression
        self.plot_evaluation_progression()
        
        # 3. Time Analysis
        self.plot_time_analysis()
        
        print("\nPerformance graphs saved to 'performance_graphs' directory")

    def plot_outcome_rates(self):
        """Plot win/draw/loss rates for each algorithm and parameter set"""
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle('Algorithm Performance: Win/Draw/Loss Rates', fontsize=16)
        
        for idx, algorithm in enumerate(["alpha-beta", "evolutionary", "pso"]):
            ax = axs[idx]
            param_names = list(self.results[algorithm].keys())
            wins = []
            draws = []
            losses = []
            
            for param in param_names:
                total = self.results[algorithm][param]["wins"] + \
                        self.results[algorithm][param]["draws"] + \
                        self.results[algorithm][param]["losses"]
                
                if total == 0:
                    continue
                    
                wins.append(self.results[algorithm][param]["wins"] / total * 100)
                draws.append(self.results[algorithm][param]["draws"] / total * 100)
                losses.append(self.results[algorithm][param]["losses"] / total * 100)
            
            if not wins:
                continue
                
            bar_width = 0.25
            index = np.arange(len(wins))
            
            ax.bar(index, wins, bar_width, label='Wins', color='#4caf50')
            ax.bar(index + bar_width, draws, bar_width, label='Draws', color='#ffc107')
            ax.bar(index + 2 * bar_width, losses, bar_width, label='Losses', color='#f44336')
            
            ax.set_xlabel('Parameter Settings')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'{algorithm.capitalize()} Algorithm')
            ax.set_xticks(index + bar_width)
            ax.set_xticklabels(param_names, rotation=15, ha='right')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('performance_graphs/outcome_rates.png', dpi=300)
        plt.close()

    def plot_evaluation_progression(self):
        """Plot average evaluation progression throughout games"""
        plt.figure(figsize=(12, 8))
        
        for algorithm in ["alpha-beta", "evolutionary", "pso"]:
            for param_name in self.results[algorithm].keys():
                all_evals = self.results[algorithm][param_name]["evaluation_data"]
                
                if not all_evals:
                    continue
                    
                # Find the longest game in this parameter set
                max_moves = max(len(evals) for evals in all_evals if evals)
                
                if max_moves == 0:
                    continue
                    
                # Create a matrix to hold interpolated evaluations
                interpolated_evals = []
                
                for evals in all_evals:
                    if not evals:
                        continue
                    # Interpolate to max_moves length
                    x_orig = np.linspace(0, 1, len(evals))
                    x_new = np.linspace(0, 1, max_moves)
                    interpolated = np.interp(x_new, x_orig, evals)
                    interpolated_evals.append(interpolated)
                
                if not interpolated_evals:
                    continue
                    
                # Calculate mean across games
                mean_evals = np.mean(interpolated_evals, axis=0)
                
                # Plot
                moves = np.arange(max_moves)
                plt.plot(moves, mean_evals, label=f"{algorithm} - {param_name}")
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Move Number')
        plt.ylabel('Stockfish Evaluation (pawns)')
        plt.title('Average Evaluation Progression Throughout Games')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('performance_graphs/evaluation_progression.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_time_analysis(self):
        """Plot move time analysis for each algorithm"""
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Move Time Analysis', fontsize=16)
        
        for idx, algorithm in enumerate(["alpha-beta", "evolutionary", "pso"]):
            ax = axs[idx]
            param_names = list(self.results[algorithm].keys())
            
            avg_times = []
            param_labels = []
            
            for param in param_names:
                all_times = self.results[algorithm][param]["time_data"]
                if not all_times:
                    continue
                
                # Flatten the list of times
                flat_times = [t for sublist in all_times for t in sublist]
                
                if not flat_times:
                    continue
                    
                avg_times.append(np.mean(flat_times))
                param_labels.append(param)
            
            if not avg_times:
                continue
                
            # Plot bar chart
            index = np.arange(len(avg_times))
            ax.bar(index, avg_times, color='#2196f3')
            
            ax.set_xlabel('Parameter Settings')
            ax.set_ylabel('Average Time (seconds)')
            ax.set_title(f'{algorithm.capitalize()} Algorithm')
            ax.set_xticks(index)
            ax.set_xticklabels(param_labels, rotation=15, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_yscale('log')  # Use log scale for better visualization
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('performance_graphs/time_analysis.png', dpi=300)
        plt.close()


if __name__ == "__main__":
    # Configuration - Update with your Stockfish path
    stockfish_path = "stockfish"  # On Linux/Mac, if in PATH
    # stockfish_path = "C:/Path/To/stockfish.exe"  # On Windows

    tester = ChessAlgorithmTester(
        stockfish_path=stockfish_path,
        num_games=1,  # Reduced for testing
        max_moves=50   # Max moves per game to limit test time
    )
    
    # Run the performance tests
    tester.run_all_tests()
    
    # Generate performance graphs
    tester.plot_performance()
    
    print("\nPerformance graphs saved to 'performance_graphs' directory")