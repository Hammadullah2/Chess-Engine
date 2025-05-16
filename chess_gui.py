import pygame
import os
import threading
import time
from chess_engine import Chess

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.SQUARE_SIZE = 80
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        self.MARGIN = 40  # Margin for coordinates
        self.TOP_MARGIN = 20  # Added top margin
        self.WINDOW_SIZE = (self.BOARD_SIZE + 2*self.MARGIN + 80, self.BOARD_SIZE + self.MARGIN + self.TOP_MARGIN + 140)
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Chess Game")
        
        # Initialize selection variables
        self.player_color = None
        self.ai_algorithm = None
        self.setup_complete = False
        
        # Run setup screen first
        self.show_setup_screen()
        
        # Initialize the chess engine
        self.chess_game = Chess()
        
        # Set up AI variables
        self.is_player_turn = self.player_color == "white"
        
        # Load images
        self.load_images()
        
        # Rest of initialization
        self.selected_piece = None
        self.dragging = False
        self.drag_pos = None
        self.last_move = None
        
        # AI suggestion variables
        self.alpha_beta_suggestion = None
        self.evolutionary_suggestion = None
        self.pso_suggestion = None
        self.suggestion_lock = threading.Lock()
        
        # Start AI suggestion threads
        self.running = True
        self.alpha_beta_thread = threading.Thread(target=self.update_alpha_beta_suggestions)
        self.evolutionary_thread = threading.Thread(target=self.update_evolutionary_suggestions)
        self.pso_thread = threading.Thread(target=self.update_pso_suggestions)
        self.alpha_beta_thread.daemon = True
        self.evolutionary_thread.daemon = True
        self.pso_thread.daemon = True
        self.alpha_beta_thread.start()
        self.evolutionary_thread.start()
        self.pso_thread.start()

        # Add thinking state
        self.ai_thinking = False
        self.thinking_dots = 0  # For animated thinking indicator
        self.thinking_timer = 0  # For thinking animation

    def show_setup_screen(self):
        """Show the initial setup screen for color and algorithm selection"""
        # Create buttons
        button_width = 200
        button_height = 50
        padding = 20
        
        # Calculate positions
        center_x = self.WINDOW_SIZE[0] // 2
        center_y = self.WINDOW_SIZE[1] // 2
        
        # Color selection buttons
        white_button = pygame.Rect(center_x - button_width - padding, 
                                 center_y - button_height - padding,
                                 button_width, button_height)
        black_button = pygame.Rect(center_x + padding,
                                 center_y - button_height - padding,
                                 button_width, button_height)
        
        # Algorithm selection buttons
        alpha_beta_button = pygame.Rect(center_x - button_width - padding,
                                      center_y + padding,
                                      button_width, button_height)
        evolutionary_button = pygame.Rect(center_x + padding,
                                        center_y + padding,
                                        button_width, button_height)
        pso_button = pygame.Rect(center_x - button_width//2,
                                center_y + button_height + 2*padding,
                                button_width, button_height)
        
        # Setup font
        font = pygame.font.Font(None, 32)
        
        while not self.setup_complete:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    
                    # Color selection
                    if white_button.collidepoint(mouse_pos):
                        self.player_color = "white"
                    elif black_button.collidepoint(mouse_pos):
                        self.player_color = "black"
                        
                    # Algorithm selection
                    if alpha_beta_button.collidepoint(mouse_pos):
                        self.ai_algorithm = "alpha-beta"
                    elif evolutionary_button.collidepoint(mouse_pos):
                        self.ai_algorithm = "evolutionary"
                    elif pso_button.collidepoint(mouse_pos):
                        self.ai_algorithm = "pso"
                    
                    # Check if setup is complete
                    if self.player_color and self.ai_algorithm:
                        self.setup_complete = True
            
            # Draw setup screen
            self.screen.fill((255, 255, 255))
            
            # Draw title
            title = font.render("Chess Game Setup", True, (0, 0, 0))
            self.screen.blit(title, (center_x - title.get_width()//2, 50))
            
            # Draw color selection text
            color_text = font.render("Select your color:", True, (0, 0, 0))
            self.screen.blit(color_text, (center_x - color_text.get_width()//2, 
                                        center_y - button_height - 2*padding - 40))
            
            # Draw algorithm selection text
            algo_text = font.render("Select AI algorithm:", True, (0, 0, 0))
            self.screen.blit(algo_text, (center_x - algo_text.get_width()//2,
                                       center_y - 20))
            
            # Draw buttons
            button_colors = {
                None: (200, 200, 200),
                "white": (100, 255, 100),
                "black": (100, 255, 100),
                "alpha-beta": (100, 255, 100),
                "evolutionary": (100, 255, 100),
                "pso": (100, 255, 100)
            }
            
            # Draw color buttons
            pygame.draw.rect(self.screen, 
                           button_colors["white" if self.player_color == "white" else None],
                           white_button)
            pygame.draw.rect(self.screen,
                           button_colors["black" if self.player_color == "black" else None],
                           black_button)
            
            # Draw algorithm buttons
            pygame.draw.rect(self.screen,
                           button_colors["alpha-beta" if self.ai_algorithm == "alpha-beta" else None],
                           alpha_beta_button)
            pygame.draw.rect(self.screen,
                           button_colors["evolutionary" if self.ai_algorithm == "evolutionary" else None],
                           evolutionary_button)
            pygame.draw.rect(self.screen,
                           button_colors["pso" if self.ai_algorithm == "pso" else None],
                           pso_button)
            
            # Draw button text
            white_text = font.render("White", True, (0, 0, 0))
            black_text = font.render("Black", True, (0, 0, 0))
            alpha_beta_text = font.render("Alpha-Beta", True, (0, 0, 0))
            evolutionary_text = font.render("Evolutionary", True, (0, 0, 0))
            pso_text = font.render("PSO", True, (0, 0, 0))
            
            self.screen.blit(white_text, (white_button.centerx - white_text.get_width()//2,
                                        white_button.centery - white_text.get_height()//2))
            self.screen.blit(black_text, (black_button.centerx - black_text.get_width()//2,
                                        black_button.centery - black_text.get_height()//2))
            self.screen.blit(alpha_beta_text, (alpha_beta_button.centerx - alpha_beta_text.get_width()//2,
                                             alpha_beta_button.centery - alpha_beta_text.get_height()//2))
            self.screen.blit(evolutionary_text, (evolutionary_button.centerx - evolutionary_text.get_width()//2,
                                               evolutionary_button.centery - evolutionary_text.get_height()//2))
            self.screen.blit(pso_text, (pso_button.centerx - pso_text.get_width()//2,
                                      pso_button.centery - pso_text.get_height()//2))
            
            pygame.display.flip()

    def make_ai_move(self):
        """Make move based on selected AI algorithm with increased computation time"""
        self.ai_thinking = True
        move = None
        
        # Run AI calculation in a separate thread to avoid freezing the UI
        def calculate_move():
            nonlocal move
            if self.ai_algorithm == "alpha-beta":
                move = self.chess_game.get_alpha_beta_move(depth=5)  # Increased depth
            elif self.ai_algorithm == "evolutionary":
                move = self.chess_game.evolutionary_algorithm(population_size=30, generations=10)  # Increased population and generations
            else:  # PSO
                move = self.chess_game.particle_swarm_optimization(num_particles=30, iterations=15)  # Increased particles and iterations
            
            # Add minimum thinking time of 2 seconds
            time.sleep(2)
            self.ai_thinking = False
        
        # Start calculation thread
        calc_thread = threading.Thread(target=calculate_move)
        calc_thread.daemon = True
        calc_thread.start()
        
        # Wait for calculation to complete while keeping UI responsive
        while self.ai_thinking:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return False
            
            # Keep UI updated while thinking
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_pieces()
            self.draw_suggestions()
            self.draw_evaluation_bar()
            self.draw_thinking_indicator()
            pygame.display.flip()
            time.sleep(0.05)  # Small delay to prevent high CPU usage
        
        if move and move != ("No move", "No move"):
            if self.chess_game.move(move[0], move[1]):
                self.last_move = move
                return True
        return False

    def load_images(self):
        """Load piece images from the 'pieces' folder"""
        self.pieces = {}
        piece_chars = {'P': 'white-pawn', 'R': 'white-rook', 'N': 'white-knight',
                      'B': 'white-bishop', 'Q': 'white-queen', 'K': 'white-king',
                      'p': 'black-pawn', 'r': 'black-rook', 'n': 'black-knight',
                      'b': 'black-bishop', 'q': 'black-queen', 'k': 'black-king'}
        
        try:
            for piece, filename in piece_chars.items():
                image_path = os.path.join('pieces', f'{filename}.png')
                self.pieces[piece] = pygame.transform.scale(
                    pygame.image.load(image_path),
                    (self.SQUARE_SIZE, self.SQUARE_SIZE)
                )
        except pygame.error as e:
            print(f"Couldn't load piece images: {e}")
            print("Make sure you have a 'pieces' folder with the required PNG images")
            exit(1)

    def get_square_from_pos(self, pos):
        """Convert screen position to board coordinates"""
        x, y = pos
        file = (x - self.MARGIN) // self.SQUARE_SIZE
        rank = (y - self.TOP_MARGIN) // self.SQUARE_SIZE
        if 0 <= file < 8 and 0 <= rank < 8:
            return f"{chr(file + 97)}{8 - rank}"
        return None

    def get_piece_at_pos(self, pos):
        """Get the piece at the given screen position"""
        square = self.get_square_from_pos(pos)
        if square:
            board_pos = self.chess_game.board_2_array(square)
            if board_pos:
                piece = self.chess_game.board[board_pos[1]][board_pos[0]]
                # Only allow selecting pieces of the current player's color
                if piece != 0 and ((self.chess_game.p_move == 1 and piece > 0) or 
                                 (self.chess_game.p_move == -1 and piece < 0)):
                    return square
        return None

    def draw_board(self):
        """Draw the chess board with coordinates"""
        self.SQUARE_SIZE = min((self.screen.get_width() - 2 * self.MARGIN - 80) // 8, (self.screen.get_height() - self.MARGIN - self.TOP_MARGIN - 140) // 8)
        self.BOARD_SIZE = self.SQUARE_SIZE * 8

        # Draw coordinates
        font = pygame.font.Font(None, 36)
        
        # Draw rank numbers (1-8)
        for rank in range(8):
            text = font.render(str(8 - rank), True, (0, 0, 0))
            self.screen.blit(text, (self.MARGIN//2 - text.get_width()//2, 
                                  self.TOP_MARGIN + rank * self.SQUARE_SIZE + self.SQUARE_SIZE//2 - text.get_height()//2))
            
        # Draw file letters (a-h)
        for file in range(8):
            text = font.render(chr(file + 97), True, (0, 0, 0))
            self.screen.blit(text, (self.MARGIN + file * self.SQUARE_SIZE + self.SQUARE_SIZE//2 - text.get_width()//2,
                                  self.TOP_MARGIN + 8 * self.SQUARE_SIZE + 5))
        
        # Draw the board squares
        for rank in range(8):
            for file in range(8):
                color = (255, 206, 158) if (rank + file) % 2 == 0 else (209, 139, 71)
                rect = pygame.Rect(self.MARGIN + file * self.SQUARE_SIZE, 
                                 self.TOP_MARGIN + rank * self.SQUARE_SIZE,
                                 self.SQUARE_SIZE, 
                                 self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight last move if exists
                if self.last_move:
                    start_pos = self.chess_game.board_2_array(self.last_move[0])
                    end_pos = self.chess_game.board_2_array(self.last_move[1])
                    if (file, rank) in [start_pos, end_pos]:
                        highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                        highlight.set_alpha(128)
                        highlight.fill((255, 255, 0))
                        self.screen.blit(highlight, (self.MARGIN + file * self.SQUARE_SIZE, self.TOP_MARGIN + rank * self.SQUARE_SIZE))

    def draw_pieces(self):
        """Draw all pieces on the board"""
        for rank in range(8):
            for file in range(8):
                piece = self.chess_game.board[rank][file]
                if piece != 0:
                    piece_char = self.get_piece_char(piece)
                    if piece_char in self.pieces and (file, rank) != self.selected_piece:
                        self.screen.blit(self.pieces[piece_char],
                                       (self.MARGIN + file * self.SQUARE_SIZE,
                                        self.TOP_MARGIN + rank * self.SQUARE_SIZE))

    def get_piece_char(self, piece):
        """Convert piece number to character representation"""
        piece_chars = {1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
                      -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k'}
        return piece_chars.get(piece, '')

    def draw_suggestions(self):
        """Draw AI move suggestions"""
        # Draw background for suggestions
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (0, self.BOARD_SIZE + self.MARGIN + self.TOP_MARGIN, 
                         self.BOARD_SIZE + 2*self.MARGIN, 140))  # Adjusted height
        
        # Draw suggestions text
        font = pygame.font.Font(None, 24)
        
        with self.suggestion_lock:
            if not self.alpha_beta_suggestion:
                self.alpha_beta_suggestion = ("No suggestion", "No suggestion")
            if not self.evolutionary_suggestion:
                self.evolutionary_suggestion = ("No suggestion", "No suggestion")
            if not self.pso_suggestion:
                self.pso_suggestion = ("No suggestion", "No suggestion")

            suggestion_texts = [
                f"Alpha-Beta: {self.alpha_beta_suggestion[0]} → {self.alpha_beta_suggestion[1]}",
                f"Evolutionary: {self.evolutionary_suggestion[0]} → {self.evolutionary_suggestion[1]}",
                f"PSO: {self.pso_suggestion[0]} → {self.pso_suggestion[1]}"
            ]

            for i, text in enumerate(suggestion_texts):
                text_surface = font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, self.BOARD_SIZE + self.MARGIN + self.TOP_MARGIN + 20 + i * 40))

    def draw_evaluation_bar(self):
        """Draw the evaluation bar on the side"""
        bar_width = 40
        bar_height = self.BOARD_SIZE

        # Calculate evaluation based on material difference
        total_eval = 0
        piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # King not counted in material
        for y in range(8):
            for x in range(8):
                piece = self.chess_game.board[y][x]
                if piece != 0:
                    sign = 1 if piece > 0 else -1
                    total_eval += piece_values[abs(piece)] * sign

        # Normalize evaluation to 0-1 range and invert it so white winning is at bottom
        max_material = 39  # Maximum material difference (excluding kings)
        normalized_eval = 1 - ((total_eval + max_material) / (2 * max_material))  # Inverted
        normalized_eval = max(0, min(1, normalized_eval))  # Clamp between 0 and 1

        # Draw the black section (top)
        black_height = int(normalized_eval * bar_height)
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (self.BOARD_SIZE + self.MARGIN, self.TOP_MARGIN, 
                         bar_width, black_height))

        # Draw the white section (bottom)
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (self.BOARD_SIZE + self.MARGIN, self.TOP_MARGIN + black_height,
                         bar_width, bar_height - black_height))

        # Draw border
        pygame.draw.rect(self.screen, (128, 128, 128), 
                        (self.BOARD_SIZE + self.MARGIN, self.TOP_MARGIN,
                         bar_width, bar_height), 1)

        # Draw evaluation score
        font = pygame.font.Font(None, 24)
        score_text = f"{total_eval:+.1f}"
        text_surface = font.render(score_text, True, (0, 0, 0))
        self.screen.blit(text_surface, 
                        (self.BOARD_SIZE + self.MARGIN + bar_width + 5,
                         self.TOP_MARGIN + bar_height//2 - text_surface.get_height()//2))

    def update_alpha_beta_suggestions(self):
        """Continuously update alpha-beta pruning suggestions"""
        while self.running:
            if not self.dragging:
                suggestion = self.chess_game.get_alpha_beta_move()
                with self.suggestion_lock:
                    self.alpha_beta_suggestion = suggestion
            time.sleep(1)  # Update every second

    def update_evolutionary_suggestions(self):
        """Continuously update evolutionary algorithm suggestions"""
        while self.running:
            if not self.dragging:
                suggestion = self.chess_game.evolutionary_algorithm()
                with self.suggestion_lock:
                    self.evolutionary_suggestion = suggestion
            time.sleep(1)  # Update every second

    def update_pso_suggestions(self):
        """Continuously update PSO suggestions"""
        while self.running:
            if not self.dragging:
                suggestion = self.chess_game.particle_swarm_optimization()
                with self.suggestion_lock:
                    self.pso_suggestion = suggestion
            time.sleep(1)  # Update every second

    def update_stockfish_evaluation(self):
        """Placeholder for stockfish evaluation"""
        while self.running:
            time.sleep(1)

    def draw_thinking_indicator(self):
        """Draw an animated 'Thinking...' indicator when AI is calculating"""
        if self.ai_thinking:
            font = pygame.font.Font(None, 36)
            dots = "." * ((self.thinking_dots % 3) + 1)
            text = font.render(f"{self.ai_algorithm} thinking{dots}", True, (0, 0, 0))
            
            # Draw with background
            bg_rect = pygame.Rect(10, 10, text.get_width() + 20, text.get_height() + 10)
            pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)
            self.screen.blit(text, (20, 15))
            
            # Update animation
            if pygame.time.get_ticks() - self.thinking_timer > 500:  # Change dots every 500ms
                self.thinking_dots += 1
                self.thinking_timer = pygame.time.get_ticks()

    def run(self):
        """Main game loop"""
        running = True
        
        # Make first move if AI is playing white
        if not self.is_player_turn:
            self.make_ai_move()
            self.is_player_turn = True
            
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.running = False
                
                # Only process mouse events during player's turn
                if self.is_player_turn:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            pos = event.pos
                            square = self.get_piece_at_pos(pos)
                            if square:
                                self.selected_piece = square
                                self.dragging = True
                                self.drag_pos = pos
                    
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1 and self.dragging:
                            pos = event.pos
                            target_square = self.get_square_from_pos(pos)
                            if target_square and self.selected_piece:
                                if self.chess_game.move(self.selected_piece, target_square):
                                    self.last_move = (self.selected_piece, target_square)
                                    self.is_player_turn = False  # Switch to AI's turn
                            
                            self.selected_piece = None
                            self.dragging = False
                    
                    elif event.type == pygame.MOUSEMOTION:
                        self.drag_pos = event.pos

            # If it's AI's turn, make a move
            if not self.is_player_turn:
                if self.make_ai_move():
                    self.is_player_turn = True  # Switch back to player's turn

            # Check for game end
            state = self.chess_game.is_end()
            if sum(state) > 0:
                print("\n*********************")
                print("      GAME OVER      ")
                print("*********************\n")
                if state == [0, 0, 1]:
                    print("BLACK WINS\n")
                elif state == [1, 0, 0]:
                    print("WHITE WINS\n")
                else:
                    print("TIE GAME\n")
                running = False
                break

            # Draw game state
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_pieces()
            
            if self.dragging and self.selected_piece:
                board_pos = self.chess_game.board_2_array(self.selected_piece)
                piece = self.chess_game.board[board_pos[1]][board_pos[0]]
                piece_char = self.get_piece_char(piece)
                if piece_char in self.pieces:
                    x, y = self.drag_pos
                    self.screen.blit(self.pieces[piece_char],
                                   (x - self.SQUARE_SIZE//2, y - self.SQUARE_SIZE//2))
            
            self.draw_suggestions()
            self.draw_evaluation_bar()
            if self.ai_thinking:
                self.draw_thinking_indicator()
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()