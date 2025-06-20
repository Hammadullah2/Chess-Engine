from copy import deepcopy
import random

"""
Comparing Evolutionary Algorithms to Alpha-Beta pruning in chess
"""
class Chess:

    def __init__(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] #Board x representation
        self.y = ['8', '7', '6', '5', '4', '3', '2', '1'] #Board y representation
        self.notation = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6} #Map of notation to part number
        self.parts = {1:'Pawn', 2:'Knight', 3:'Bishop', 4:'Rook', 5:'Queen', 6:'King'} #Map of number to part
        self.c_escape = {} #Possible check escapes
        self.reset(EPD=EPD) #Reset game board and state


    def reset(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.log = [] #Game log
        self.init_pos = EPD #Inital position
        self.EPD_table = {} #EPD hashtable
        self.p_move = 1 #Current players move white = 1 black = -1
        self.castling = [1, 1, 1, 1] #Castling control
        self.en_passant = None #En passant control
        self.prev_move = None #Previous move
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]] #Generate empty chess board
        self.load_EPD(EPD) #Load in game starting position


    def display(self):
        result = '  a b c d e f g h  \n  ----------------\n'
        for c, y in enumerate(self.board):
            result += f'{8-c}|'
            for x in y:
                if x != 0:
                    n = getattr(Chess, self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.lower() if x < 0 else getattr(Chess, self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.upper()
                    if n == '':
                        n = 'p' if x < 0 else 'P'
                    result += n
                else:
                    result += '.'
                result += ' '
            result += f'|{8-c}\n'
        result += '  ----------------\n  a b c d e f g h\n'
        print(result)


    def board_2_array(self, cord):
        cord = list(cord)
        if len(cord) == 2 and str(cord[0]).lower() in self.x and str(cord[1]) in self.y:
            return self.x.index(str(cord[0]).lower()), self.y.index(str(cord[1]))
        else:
            return None


    def EPD_hash(self):
        result = ''
        for i, rank in enumerate(self.board):
            e_count = 0
            for square in rank:
                if square == 0:
                    e_count += 1
                else:
                    if e_count > 0:
                        result += str(e_count)
                    e_count = 0
                    p_name = self.parts[int(square) if square > 0 else int(square)*(-1)] #Get name of part
                    p_notation = getattr(Chess, p_name)().notation
                    if p_notation == '':
                        p_notation = 'p'
                    if square < 0:
                        p_notation = str(p_notation).lower()
                    else:
                        p_notation = str(p_notation).upper()
                    result += p_notation
            if e_count > 0:
                result += str(e_count)
            if i < 7:
                result += '/'
        if self.p_move == -1:
            result += ' w'
        else:
            result += ' b'
        result += ' '
        if sum(self.castling) == 0:
            result += '-'
        else:
            if self.castling[0] == 1:
                result += 'K'
            if self.castling[1] == 1:
                result += 'Q'
            if self.castling[2] == 1:
                result += 'k'
            if self.castling[3] == 1:
                result += 'q'
        result += ' '
        if self.en_passant == None:
            result += '-'
        else:
            result += f'{self.x[self.en_passant[0]]}{self.y[self.en_passant[1]]}'
        return result


    def load_EPD(self, EPD):
        data = EPD.split(' ')
        if len(data) == 4:
            for x, rank in enumerate(data[0].split('/')):
                y = 0
                for p in rank:
                    if p.isdigit():
                        for i in range(int(p)):
                            self.board[x][y] = 0
                            y += 1
                    else:
                        self.board[x][y] = self.notation[str(p).lower()]*(-1) if str(p).islower() else self.notation[str(p).lower()]
                        y += 1
            self.p_move = 1 if data[1] == 'w' else -1
            if 'K' in data[2]:
                self.castling[0] = 1
            else:
                self.castling[0] = 0
            if 'Q' in data[2]:
                self.castling[1] = 1
            else:
                self.castling[1] = 0
            if 'k' in data[2]:
                self.castling[2] = 1
            else:
                self.castling[2] = 0
            if 'q' in data[2]:
                self.castling[3] = 1
            else:
                self.castling[3] = 0
            self.en_passant = None if data[3] == '-' else self.board_2_array(data[3])
            return True
        else:
            return False


    def log_move(self, part, cur_cord, next_cord, cur_pos, next_pos, n_part=None):
        #to remove ambiguity where multiple pieces could make the move add starting identifier after piece notation ex Rab8
        if part == 6*self.p_move and next_pos[0]-cur_pos[0] == 2:
            move = '0-0'
        elif part == 6*self.p_move and next_pos[0]-cur_pos[0] == -2:
            move = '0-0-0'
        elif part == 1*self.p_move and n_part != None:
            move = f'{str(next_cord).lower()}={str(n_part).upper()}'
        else:
            p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
            move = str(getattr(Chess, p_name)().notation).upper() #Get part notation
            if self.board[next_pos[1]][next_pos[0]] != 0 or (next_pos == self.en_passant and (part == 1 or part == -1)): #Check if there is a capture
                move += 'x' if move != '' else str(cur_cord)[0] + 'x'
            move += str(next_cord).lower()
        self.log.append(move)


    def move(self, cur_pos, next_pos, promotion=None):
        cp = self.board_2_array(cur_pos)
        np = self.board_2_array(next_pos)
        if self.valid_move(cp, np) == True:
            part = self.board[cp[1]][cp[0]]
            if np == self.en_passant and (part == 1 or part == -1):
                self.board[self.en_passant[1]-(self.p_move*(-1))][self.en_passant[0]] = 0
            self.log_move(part, cur_pos, next_pos, cp, np, n_part=promotion)
            self.prev_move = self.board
            if (part == 1 and np[1] == 4) or (part == -1 and np[1] == 3):
                self.en_passant = (np[0], np[1]+1) if part == 1 else (np[0], np[1]-1)
            elif part == 6*self.p_move and np[0]-cp[0] == 2:
                self.board[np[1]][np[0]-1] = 4*self.p_move
                self.board[np[1]][np[0]+1] = 0
            elif part == 6*self.p_move and np[0]-cp[0] == -2:
                self.board[np[1]][np[0]+1] = 4*self.p_move
                self.board[np[1]][np[0]-2] = 0
            else:
                self.en_passant = None
            if part == 6*self.p_move:
                if self.p_move == 1:
                    self.castling[0] = 0
                    self.castling[1] = 0
                else:
                    self.castling[2] = 0
                    self.castling[3] = 0
            elif part == 4*self.p_move:
                if self.p_move == 1:
                    if cp == (0, 7):
                        self.castling[1] = 0
                    else:
                        self.castling[0] = 0
                else:
                    if cp == (0, 0):
                        self.castling[3] = 0
                    else:
                        self.castling[2] = 0
            self.board[cp[1]][cp[0]] = 0
            
            # Handle pawn promotion
            if (part == 1 and np[1] == 0) or (part == -1 and np[1] == 7):
                if promotion is None:
                    promotion = 'q'  # Default to queen
                promotion_piece = self.notation[promotion] * self.p_move
                self.board[np[1]][np[0]] = promotion_piece
            else:
                self.board[np[1]][np[0]] = part
                
            hash = self.EPD_hash()
            if hash in self.EPD_table:
                self.EPD_table[hash] += 1
            else:
                self.EPD_table[hash] = 1
            self.p_move = self.p_move * (-1)
            return True
        return False


    def valid_move(self, cur_pos, next_pos):
        """Determine if player move is valid game move"""
        if cur_pos != None and next_pos != None:
            part = self.board[cur_pos[1]][cur_pos[0]]
            if part * self.p_move > 0 and part != 0:
                p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
                v_moves = getattr(Chess, p_name).movement(self, self.p_move, cur_pos, capture=True)
                
                if next_pos in v_moves:
                    # Make temporary move to check if it puts own king in check
                    temp_board = deepcopy(self)
                    # Handle pawn promotion in temp board
                    if (part == 1 and next_pos[1] == 0) or (part == -1 and next_pos[1] == 7):
                        temp_board.board[next_pos[1]][next_pos[0]] = self.notation['q'] * self.p_move
                    else:
                        temp_board.board[next_pos[1]][next_pos[0]] = temp_board.board[cur_pos[1]][cur_pos[0]]
                    temp_board.board[cur_pos[1]][cur_pos[0]] = 0
                    
                    # Find king's position
                    king_pos = None
                    for y in range(8):
                        for x in range(8):
                            if temp_board.board[y][x] == 6 * self.p_move:
                                king_pos = (x, y)
                                break
                        if king_pos:
                            break
                    
                    # Check if any opponent piece can capture the king
                    if king_pos:
                        for y in range(8):
                            for x in range(8):
                                opp_piece = temp_board.board[y][x]
                                if opp_piece * self.p_move < 0:  # opponent's piece
                                    opp_name = self.parts[abs(opp_piece)]
                                    opp_moves = getattr(Chess, opp_name).movement(temp_board, -self.p_move, (x, y), capture=True)
                                    if king_pos in opp_moves:  # King is in check
                                        return False
                    return True
        return False


    def possible_board_moves(self, capture=True):
        moves = {}
        for y, row in enumerate(self.board):
            for x, part in enumerate(row):
                if part != 0:
                    p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
                    p_colour = 1 if part > 0 else -1
                    v_moves = getattr(Chess, p_name).movement(self, p_colour, [x, y], capture=capture)
                    if len(self.log) > 0 and '+' in self.log[-1]:
                        v_moves = [m for m in v_moves if (x, y) in self.c_escape and m in self.c_escape[(x, y)]]
                    moves[f'{str(self.x[x]).upper() if p_colour > 0 else str(self.x[x]).lower()}{self.y[y]}'] = v_moves
        return moves


    def is_checkmate(self, moves):
        """Determine if the current game state results in a check mate or not"""
        self.c_escape = {}
        k_pos = None  # King position
        
        # First find the king's position
        for y, row in enumerate(self.board):
            for x, piece in enumerate(row):
                if piece == self.King().value * (self.p_move):  # Current player's king
                    k_pos = (x, y)
                    break
            if k_pos:
                break
                
        if not k_pos:
            return [1, 0, 0] if self.p_move == -1 else [0, 0, 1]  # No king found
            
        # Check if king is in check
        is_in_check = False
        for y, row in enumerate(self.board):
            for x, piece in enumerate(row):
                if piece * self.p_move < 0:  # Enemy piece
                    piece_type = self.parts[abs(piece)]
                    enemy_moves = getattr(Chess, piece_type).movement(self, -self.p_move, (x, y), capture=True)
                    if k_pos in enemy_moves:
                        is_in_check = True
                        break
            if is_in_check:
                break
                
        if not is_in_check:
            return [0, 0, 0]  # Not in check, so not checkmate
            
        # If in check, see if any move can get out of check
        for start_square, possible_moves in moves.items():
            if ((self.p_move == 1 and start_square[0].isupper()) or 
                (self.p_move == -1 and start_square[0].islower())):
                start_pos = self.board_2_array(start_square)
                for move in possible_moves:
                    # Try the move on a temporary board
                    temp_board = deepcopy(self)
                    temp_board.move(start_square, f"{self.x[move[0]]}{self.y[move[1]]}")
                    temp_board.p_move *= -1  # Switch back to original player
                    
                    # After move, check if king is still in check
                    still_in_check = False
                    new_k_pos = k_pos
                    if start_pos == k_pos:
                        new_k_pos = move  # King moved
                        
                    for y, row in enumerate(temp_board.board):
                        for x, piece in enumerate(row):
                            if piece * self.p_move < 0:  # Enemy piece
                                piece_type = self.parts[abs(piece)]
                                enemy_moves = getattr(Chess, piece_type).movement(temp_board, -self.p_move, (x, y), capture=True)
                                if new_k_pos in enemy_moves:
                                    still_in_check = True
                                    break
                        if still_in_check:
                            break
                            
                    if not still_in_check:
                        return [0, 0, 0]  # Found an escape move, not checkmate
                        
        # No escape moves found, it's checkmate
        if self.p_move == 1:
            return [0, 0, 1]  # Black wins
        else:
            return [1, 0, 0]  # White wins


    def pawn_promotion(self, n_part=None):
        if n_part == None:
            while True:
                n_part = input('\nPawn Promotion - What peice would you like to switch too:\n\n*Queen[q]\n*Bishop[b]\n*Knight[n]\n*Rook[r]\n')
                if str(n_part).lower() not in ['q', 'b', 'n', 'r', 'queen', 'bishop', 'knight', 'rook']:
                    print('\nInvalid Option')
                else:
                    break
            if len(n_part) > 1:
                n_part = getattr(Chess, str(n_part).capitalize())().notation
        part = self.notation[str(n_part).lower()]*self.p_move
        pos = self.board_2_array(self.log[-1].replace('+', '').split('x')[-1])
        if pos != None:
            self.board[pos[1]][pos[0]] = part
            self.log[-1] += f'={str(n_part).upper()}'
            return True
        else:
            return False


    def fifty_move_rule(self, moves, choice=None):
        if len(self.log) > 100:
            for m in self.log[-100:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        if choice == None:
            while True:
                choice = input('Fifty move rule - do you want to claim a draw? [Y/N]')
                if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                    return True
                elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                    return False
            print('Unsupported answer')
        if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
            return True
        elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
            return False


    def seventy_five_move_rule(self, moves):
        if len(self.log) > 150:
            for m in self.log[-150:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        return True


    def three_fold_rule(self, hash):
        if hash in self.EPD_table:
            if self.EPD_table[hash] == 3:
                while True:
                    choice = input('Three fold rule - do you want to claim a draw? [Y/N]')
                    if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                        return True
                    elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                        return False
                    print('Unsupported answer')
        return False


    def five_fold_rule(self, hash):
        if hash in self.EPD_table:
            if self.EPD_table[hash] >= 5:
                return True
        return False

    
    def is_dead_position(self, moves):
        #King and bishop against king and bishop with both bishops on squares of the same colour
        a_pieces = []
        for y in self.board:
            for x in y:
                if x != 0:
                    a_pieces.append(x)
                if len(a_pieces) > 4:
                    return False
        if len(a_pieces) == 2 and -6 in a_pieces and 6 in a_pieces:
            return True
        elif len(a_pieces) == 3 and ((-6 in a_pieces and 3 in a_pieces and 6 in a_pieces) or (-6 in a_pieces and -3 in a_pieces and 6 in a_pieces)):
            return True
        elif len(a_pieces) == 3 and ((-6 in a_pieces and 2 in a_pieces and 6 in a_pieces) or (-6 in a_pieces and -2 in a_pieces and 6 in a_pieces)):
            return True
        return False

    
    def is_stalemate(self, moves):
        if False not in [False for p, a in moves.items() if len(a) > 0 and ((self.p_move == 1 and str(p[0]).isupper()) or (self.p_move == -1 and str(p[0]).islower()))]:
            return True
        return False

    def is_draw(self, moves, hash):
        if self.is_stalemate(moves) == True:
            return True
        elif self.is_dead_position(moves) == True:
            return True
        elif self.seventy_five_move_rule(moves) == True:
            return True
        elif self.five_fold_rule(hash) == True:
            return True
        elif self.fifty_move_rule(moves) == True:
            return True
        elif self.three_fold_rule(hash) == True:
            return True
        return False

   
    def is_end(self):
        w_king = False
        b_king = False
        for y, row in enumerate(self.board):
            for x, piece in enumerate(row):
                if piece == self.King().value * (-1):
                    b_king = True
                elif piece == self.King().value:
                    w_king = True
        if w_king == False and b_king == False:
            return [0, 1, 0]
        elif w_king == False:
            return [0, 0, 1]
        elif b_king == False:
            return [1, 0, 0]
        
        # Find the current player's king position
        king_pos = None
        for y in range(8):
            for x in range(8):
                if self.board[y][x] == 6 * self.p_move:
                    king_pos = (x, y)
                    break
            if king_pos:
                break

        moves = self.possible_board_moves(capture=True)
        
        # If no legal moves for any piece:
        if not any(len(mv) > 0 for mv in moves.values()):
            # Check if king is attacked
            in_check = False
            for y, row in enumerate(self.board):
                for x, piece in enumerate(row):
                    if piece * self.p_move < 0:
                        p_name = self.parts[abs(piece)]
                        enemy_moves = getattr(Chess, p_name).movement(self, -self.p_move, (x, y), capture=True)
                        if king_pos and king_pos in enemy_moves:
                            in_check = True
                            break
                if in_check:
                    break
            if in_check:
                # Checkmate: if it is white's move, black wins; otherwise white wins.
                return [0, 0, 1] if self.p_move == 1 else [1, 0, 0]
            else:
                # Stalemate (draw)
                return [0, 1, 0]

        check_mate = self.is_checkmate(moves)
        hash = self.EPD_hash()
        if sum(check_mate) > 0:
            return check_mate
        elif self.is_draw(moves, hash) == True:
            return [0, 1, 0]
        return [0, 0, 0]

    
    def check_state(self, hash):
        if len(self.log) > 0 and self.p_move == 1 and (self.log[-1][0].isupper() == False or self.log[-1][0] == 'P') and True in [True for l in self.log[-1] if l == '8']:
            return 'PP' #Pawn promotion
        elif len(self.log) > 0 and self.p_move == -1 and (self.log[-1][0].isupper() == False or self.log[-1][0] == 'P') and True in [True for l in self.log[-1] if l == '1']:
            return 'PP' #Pawn promotion
        elif hash in self.EPD_table and self.EPD_table[hash] == 3:
            return '3F' #3 Fold
        elif len(self.log) > 100:
            for m in self.log[-100:]:
                if 'x' in m or m[0].islower():
                    return None
            return '50M' #50 move
        else:
            return None

   
    def get_alpha_beta_move(self, depth=3):
        """Get the best move using Alpha-Beta pruning with configurable depth."""
        _, best_move = self.alpha_beta(depth, float('-inf'), float('inf'), True)
        return best_move

    
    def evolutionary_algorithm(self, population_size=30, generations=10):
        """Get the best move using an Evolutionary Algorithm with improved diversity and evaluation."""
        import random
        from copy import deepcopy
        mutation_rate = 0.1
        # Gather all valid moves for the current player
        possible_moves = self.possible_board_moves()
        valid_moves = []
        for start_square, moves in possible_moves.items():
            if ((self.p_move == 1 and start_square[0].isupper()) or 
                (self.p_move == -1 and start_square[0].islower())) and moves:
                for move_coord in moves:
                    # Check if this is a pawn promotion
                    start_pos = self.board_2_array(start_square)
                    piece = self.board[start_pos[1]][start_pos[0]]
                    promotions = [None]
                    if abs(piece) == 1:  # pawn
                        if (piece > 0 and move_coord[1] == 0) or (piece < 0 and move_coord[1] == 7):
                            promotions = ['q','r','b','n']
                    
                    for promo in promotions:
                        valid_moves.append((start_square, f"{self.x[move_coord[0]]}{self.y[move_coord[1]]}", promo))
        if not valid_moves:
            return ("No move", "No move", None)
        # Initialize population with random valid moves
        population = [random.choice(valid_moves) for _ in range(population_size)]
        # Evolution process
        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for move in population:
                temp_board = deepcopy(self)
                if temp_board.move(move[0], move[1], move[2]):
                    fitness_scores.append(temp_board.evaluate_position() * self.p_move)
                else:
                    fitness_scores.append(float('-inf'))
            # Selection (tournament)
            selected = []
            for _ in range(population_size // 2):
                tournament = random.sample(list(enumerate(fitness_scores)), min(3, len(fitness_scores)))
                winner = max(tournament, key=lambda x: x[1])[0]
                selected.append(population[winner])
            # Crossover and Mutation
            new_population = selected.copy()
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                # Crossover: take move from either parent
                child = random.choice([parent1, parent2])
                # Mutation: randomly select a new move
                if random.random() < mutation_rate:
                    child = random.choice(valid_moves)
                new_population.append(child)
            population = new_population
        # Return best move from final population
        best_move = None
        best_score = float('-inf')
        for move in population:
            temp_board = deepcopy(self)
            if temp_board.move(move[0], move[1], move[2]):
                score = temp_board.evaluate_position() * self.p_move
                if score > best_score:
                    best_score = score
                    best_move = move
        return best_move if best_move else ("No move", "No move", None)

   
    def particle_swarm_optimization(self, num_particles=30, iterations=15):
        """Get the best move using PSO with improved diversity and evaluation."""
        import random
        from copy import deepcopy
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive weight
        c2 = 1.5  # Social weight
        possible_moves = self.possible_board_moves()
        valid_moves = []
        for start_square, moves in possible_moves.items():
            if ((self.p_move == 1 and start_square[0].isupper()) or 
                (self.p_move == -1 and start_square[0].islower())) and moves:
                for move in moves:
                    # Check if this is a pawn promotion
                    start_pos = self.board_2_array(start_square)
                    piece = self.board[start_pos[1]][start_pos[0]]
                    promotions = [None]
                    if abs(piece) == 1:  # pawn
                        if (piece > 0 and move[1] == 0) or (piece < 0 and move[1] == 7):
                            promotions = ['q','r','b','n']
                    
                    for promo in promotions:
                        valid_moves.append((start_square, f"{self.x[move[0]]}{self.y[move[1]]}", promo))
        if not valid_moves:
            return ("No move", "No move", None)
        # Initialize particles
        particles = [random.choice(valid_moves) for _ in range(num_particles)]
        velocities = [random.randint(-2, 2) for _ in range(num_particles)]
        personal_best_positions = particles[:]
        personal_best_scores = []
        for particle in particles:
            temp_board = deepcopy(self)
            if temp_board.move(particle[0], particle[1], particle[2]):
                score = temp_board.evaluate_position() * self.p_move
            else:
                score = float('-inf')
            personal_best_scores.append(score)
        # Initialize global best
        global_best_score = max(personal_best_scores)
        global_best_position = personal_best_positions[personal_best_scores.index(global_best_score)]
        # PSO iterations
        for _ in range(iterations):
            for i in range(num_particles):
                # Update velocity
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (valid_moves.index(personal_best_positions[i]) - valid_moves.index(particles[i]))
                social = c2 * r2 * (valid_moves.index(global_best_position) - valid_moves.index(particles[i]))
                velocities[i] = int(w * velocities[i] + cognitive + social)
                # Update position
                new_index = (valid_moves.index(particles[i]) + velocities[i]) % len(valid_moves)
                particles[i] = valid_moves[new_index]
                # Evaluate new position
                temp_board = deepcopy(self)
                if temp_board.move(particles[i][0], particles[i][1], particles[i][2]):
                    score = temp_board.evaluate_position() * self.p_move
                else:
                    score = float('-inf')
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                    # Update global best
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]
        return global_best_position

    
    def evaluate_position(self):
        """Evaluate the current board position with material and positional factors."""
        score = 0
        piece_values = {
            1: 100,   # Pawn
            2: 320,   # Knight
            3: 330,   # Bishop
            4: 500,   # Rook
            5: 900,   # Queen
            6: 20000  # King
        }
        # Piece-square tables (simplified, midgame)
        pst = {
            1: [
                [0, 5, 5, 0, 5, 10, 50, 0],
                [0, 10, -5, 0, 5, 10, 50, 0],
                [0, 10, -10, 0, 10, 20, 50, 0],
                [0, -20, 0, 20, 25, 30, 50, 0],
                [0, -20, 0, 20, 25, 30, 50, 0],
                [0, 10, -10, 0, 10, 20, 50, 0],
                [0, 10, -5, 0, 5, 10, 50, 0],
                [0, 5, 5, 0, 5, 10, 50, 0],
            ],
            2: [
                [-50, -40, -30, -30, -30, -30, -40, -50],
                [-40, -20, 0, 0, 0, 0, -20, -40],
                [-30, 0, 10, 15, 15, 10, 0, -30],
                [-30, 5, 15, 20, 20, 15, 5, -30],
                [-30, 0, 15, 20, 20, 15, 0, -30],
                [-30, 5, 10, 15, 15, 10, 5, -30],
                [-40, -20, 0, 5, 5, 0, -20, -40],
                [-50, -40, -30, -30, -30, -30, -40, -50],
            ],
            3: [
                [-20, -10, -10, -10, -10, -10, -10, -20],
                [-10, 0, 0, 0, 0, 0, 0, -10],
                [-10, 0, 5, 10, 10, 5, 0, -10],
                [-10, 5, 5, 10, 10, 5, 5, -10],
                [-10, 0, 10, 10, 10, 10, 0, -10],
                [-10, 10, 10, 10, 10, 10, 10, -10],
                [-10, 5, 0, 0, 0, 0, 5, -10],
                [-20, -10, -10, -10, -10, -10, -10, -20],
            ],
            4: [
                [0, 0, 0, 5, 5, 0, 0, 0],
                [-5, 0, 0, 0, 0, 0, 0, -5],
                [-5, 0, 0, 0, 0, 0, 0, -5],
                [-5, 0, 0, 0, 0, 0, 0, -5],
                [-5, 0, 0, 0, 0, 0, 0, -5],
                [-5, 0, 0, 0, 0, 0, 0, -5],
                [5, 10, 10, 10, 10, 10, 10, 5],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            5: [
                [-20, -10, -10, -5, -5, -10, -10, -20],
                [-10, 0, 0, 0, 0, 0, 0, -10],
                [-10, 0, 5, 5, 5, 5, 0, -10],
                [-5, 0, 5, 5, 5, 5, 0, -5],
                [0, 0, 5, 5, 5, 5, 0, -5],
                [-10, 5, 5, 5, 5, 5, 0, -10],
                [-10, 0, 5, 0, 0, 0, 0, -10],
                [-20, -10, -10, -5, -5, -10, -10, -20],
            ],
            6: [
                [-30, -40, -40, -50, -50, -40, -40, -30],
                [-30, -40, -40, -50, -50, -40, -40, -30],
                [-30, -40, -40, -50, -50, -40, -40, -30],
                [-30, -40, -40, -50, -50, -40, -40, -30],
                [-20, -30, -30, -40, -40, -30, -30, -20],
                [-10, -20, -20, -20, -20, -20, -20, -10],
                [20, 20, 0, 0, 0, 0, 20, 20],
                [20, 30, 10, 0, 0, 10, 30, 20],
            ],
        }
        for y in range(8):
            for x in range(8):
                piece = self.board[y][x]
                if piece != 0:
                    piece_type = abs(piece)
                    multiplier = 1 if piece > 0 else -1
                    # Material
                    score += piece_values[piece_type] * multiplier
                    # Piece-square table (flip for black)
                    if piece_type in pst:
                        if piece > 0:
                            score += pst[piece_type][y][x]
                        else:
                            score -= pst[piece_type][7-y][x]
        # Penalty for doubled pawns
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                if self.board[rank][file] == 1:
                    white_pawns += 1
                elif self.board[rank][file] == -1:
                    black_pawns += 1
            if white_pawns > 1:
                score -= 20 * (white_pawns - 1)
            if black_pawns > 1:
                score += 20 * (black_pawns - 1)
        # King safety (simple): bonus for castling rights
        if self.castling[0] or self.castling[1]:
            score += 30
        if self.castling[2] or self.castling[3]:
            score -= 30
        return score

    
    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        """Alpha-Beta pruning algorithm for move searching."""
        if depth == 0:
            return self.evaluate_position(), None
        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for start_square, moves in self.possible_board_moves().items():
                if (self.p_move == 1 and start_square[0].isupper()) or (self.p_move == -1 and start_square[0].islower()):
                    for move in moves:
                        # Check if this is a pawn promotion
                        start_pos = self.board_2_array(start_square)
                        piece = self.board[start_pos[1]][start_pos[0]]
                        promotions = [None]
                        if abs(piece) == 1:  # pawn
                            if (piece > 0 and move[1] == 0) or (piece < 0 and move[1] == 7):
                                promotions = ['q','r','b','n']
                        
                        for promo in promotions:
                            temp_board = deepcopy(self)
                            move_str = f"{self.x[move[0]]}{self.y[move[1]]}"
                            if temp_board.move(start_square, move_str, promotion=promo):
                                eval_score, _ = temp_board.alpha_beta(depth - 1, alpha, beta, False)
                                if eval_score > max_eval:
                                    max_eval = eval_score
                                    best_move = (start_square, move_str, promo)
                                alpha = max(alpha, eval_score)
                                if beta <= alpha:
                                    break
                            if promo is not None and beta <= alpha:
                                break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for start_square, moves in self.possible_board_moves().items():
                if (self.p_move == 1 and start_square[0].isupper()) or (self.p_move == -1 and start_square[0].islower()):
                    for move in moves:
                        # Check if this is a pawn promotion
                        start_pos = self.board_2_array(start_square)
                        piece = self.board[start_pos[1]][start_pos[0]]
                        promotions = [None]
                        if abs(piece) == 1:  # pawn
                            if (piece > 0 and move[1] == 0) or (piece < 0 and move[1] == 7):
                                promotions = ['q','r','b','n']
                        
                        for promo in promotions:
                            temp_board = deepcopy(self)
                            move_str = f"{self.x[move[0]]}{self.y[move[1]]}"
                            if temp_board.move(start_square, move_str, promotion=promo):
                                eval_score, _ = temp_board.alpha_beta(depth - 1, alpha, beta, True)
                                if eval_score < min_eval:
                                    min_eval = eval_score
                                    best_move = (start_square, move_str, promo)
                                beta = min(beta, eval_score)
                                if beta <= alpha:
                                    break
                            if promo is not None and beta <= alpha:
                                break
            return min_eval, best_move

    """
    Chess peice object for the king
    """
    class King:
        
        def __init__(self):
            self.value = 6 #Numerical value of piece
            self.notation = 'K' #Chess notation

        
        def movement(game, player, pos, capture=True):
            result = []
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+1][pos[0]]*player < 0 or game.board[pos[1]+1][pos[0]] == 0):
                result.append((pos[0], pos[1]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-1][pos[0]]*player < 0 or game.board[pos[1]-1][pos[0]] == 0):
                result.append((pos[0], pos[1]-1))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]][pos[0]+1]*player < 0 or game.board[pos[1]][pos[0]+1] == 0):
                result.append((pos[0]+1, pos[1]))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]][pos[0]-1]*player < 0 or game.board[pos[1]][pos[0]-1] == 0):
                result.append((pos[0]-1, pos[1]))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]+1][pos[0]+1]*player < 0 or game.board[pos[1]+1][pos[0]+1] == 0):
                result.append((pos[0]+1, pos[1]+1))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]+1][pos[0]-1]*player < 0 or game.board[pos[1]+1][pos[0]-1] == 0):
                result.append((pos[0]-1, pos[1]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]-1][pos[0]+1]*player < 0 or game.board[pos[1]-1][pos[0]+1] == 0):
                result.append((pos[0]+1, pos[1]-1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]-1][pos[0]-1]*player < 0 or game.board[pos[1]-1][pos[0]-1] == 0):
                result.append((pos[0]-1, pos[1]-1))
            if (pos == (4, 7) or pos == (4, 0)) and game.board[pos[1]][pos[0]+1] == 0 and game.board[pos[1]][pos[0]+2] == 0 and ((game.castling[0] == 1 and game.p_move == 1) or (game.castling[2] == 1 and game.p_move == -1)):
                result.append((pos[0]+2, pos[1]))
            if (pos == (4, 7) or pos == (4, 0)) and game.board[pos[1]][pos[0]-1] == 0 and game.board[pos[1]][pos[0]-2] == 0 and ((game.castling[1] == 1 and game.p_move == 1) or (game.castling[3] == 1 and game.p_move == -1)):
                result.append((pos[0]-2, pos[1]))
            return result

    """
    Chess peice object for the queen
    """
    class Queen:
       
        def __init__(self):
            self.value = 5 #Numerical value of piece
            self.notation = 'Q' #Chess notation

        
        def movement(game, player, pos, capture=True):
            result = []
            check = [True, True, True, True, True, True, True, True]
            for c in range(1, 8, 1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+c][pos[0]]*player < 0 or game.board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[0], pos[1]+c))
                    if game.board[pos[1]+c][pos[0]]*player < 0 and capture == True:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-c][pos[0]]*player < 0 or game.board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[0], pos[1]-c))
                    if game.board[pos[1]-c][pos[0]]*player < 0 and capture == True:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]][pos[0]+c]*player < 0 or game.board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c, pos[1]))
                    if game.board[pos[1]][pos[0]+c]*player < 0 and capture == True:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]][pos[0]-c]*player < 0 or game.board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c, pos[1]))
                    if game.board[pos[1]][pos[0]-c]*player < 0 and capture == True:
                        check[3] = False
                else:
                    check[3] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]+c][pos[0]+c]*player < 0 or game.board[pos[1]+c][pos[0]+c] == 0) and check[4] == True:
                    result.append((pos[0]+c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]+c]*player < 0 and capture == True:
                        check[4] = False
                else:
                    check[4] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]+c][pos[0]-c]*player < 0 or game.board[pos[1]+c][pos[0]-c] == 0) and check[5] == True:
                    result.append((pos[0]-c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]-c]*player < 0 and capture == True:
                        check[5] = False
                else:
                    check[5] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]-c][pos[0]+c]*player < 0 or game.board[pos[1]-c][pos[0]+c] == 0) and check[6] == True:
                    result.append((pos[0]+c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]+c]*player < 0 and capture == True:
                        check[6] = False
                else:
                    check[6] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]-c][pos[0]-c]*player < 0 or game.board[pos[1]-c][pos[0]-c] == 0) and check[7] == True:
                    result.append((pos[0]-c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]-c]*player < 0 and capture == True:
                        check[7] = False
                else:
                    check[7] = False
                if True not in check:
                    break
            return result

    """
    Chess peice object for the rook
    """
    class Rook:
        
        def __init__(self):
            self.value = 4 #Numerical value of piece
            self.notation = 'R' #Chess notation

        def movement(game, player, pos, capture=True):
            result = []
            check = [True, True, True, True]
            for c in range(1, 8, 1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+c][pos[0]]*player < 0 or game.board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[0], pos[1]+c))
                    if game.board[pos[1]+c][pos[0]]*player < 0 and capture == True:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-c][pos[0]]*player < 0 or game.board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[0], pos[1]-c))
                    if game.board[pos[1]-c][pos[0]]*player < 0 and capture == True:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]][pos[0]+c]*player < 0 or game.board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c, pos[1]))
                    if game.board[pos[1]][pos[0]+c]*player < 0 and capture == True:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]][pos[0]-c]*player < 0 or game.board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c, pos[1]))
                    if game.board[pos[1]][pos[0]-c]*player < 0 and capture == True:
                        check[3] = False
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    """
    Chess peice object for the bishop
    """
    class Bishop:
        
        def __init__(self):
            self.value = 3 #Numerical value of piece
            self.notation = 'B' #Chess notation

       
        def movement(game, player, pos, capture=True):
            result = []
            check = [True, True, True, True]
            for c in range(1, 8, 1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]+c][pos[0]+c]*player < 0 or game.board[pos[1]+c][pos[0]+c] == 0) and check[0] == True:
                    result.append((pos[0]+c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]+c]*player < 0 and capture == True:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]+c][pos[0]-c]*player < 0 or game.board[pos[1]+c][pos[0]-c] == 0) and check[1] == True:
                    result.append((pos[0]-c, pos[1]+c))
                    if game.board[pos[1]+c][pos[0]-c]*player < 0 and capture == True:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]-c][pos[0]+c]*player < 0 or game.board[pos[1]-c][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]+c]*player < 0 and capture == True:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]-c][pos[0]-c]*player < 0 or game.board[pos[1]-c][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c, pos[1]-c))
                    if game.board[pos[1]-c][pos[0]-c]*player < 0 and capture == True:
                        check[3] = False
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    """
    Chess peice object for the knight
    """
    class Knight:
        
        def __init__(self):
            self.value = 2 #Numerical value of piece
            self.notation = 'N' #Chess notation

       
        def movement(game, player, pos, capture=True):
            result = []
            for i in [-1, 1]:
                if pos[0]-i >= 0 and pos[0]-i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (game.board[pos[1]-(2*i)][pos[0]-i]*player < 0 or game.board[pos[1]-(2*i)][pos[0]-i] == 0):
                    result.append((pos[0]-i, pos[1]-(2*i)))
                if pos[0]+i >= 0 and pos[0]+i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (game.board[pos[1]-(2*i)][pos[0]+i]*player < 0 or game.board[pos[1]-(2*i)][pos[0]+i] == 0):
                    result.append((pos[0]+i, pos[1]-(2*i)))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]-i >= 0 and pos[1]-i <= 7 and (game.board[pos[1]-i][pos[0]-(2*i)]*player < 0 or game.board[pos[1]-i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i), pos[1]-i))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]+i >= 0 and pos[1]+i <= 7 and (game.board[pos[1]+i][pos[0]-(2*i)]*player < 0 or game.board[pos[1]+i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i), pos[1]+i))
            return result

    """
    Chess peice object for the pawn
    """
    class Pawn:
        
        def __init__(self):
            self.value = 1 #Numerical value of piece
            self.notation = '' #Chess notation

        
        def movement(game, player, pos, capture=True):
            result = []
            init = 1 if player < 0 else 6
            amt = 1 if pos[1] != init else 2
            for i in range(amt):
                if pos[1]-((i+1)*player) >= 0 and pos[1]-((i+1)*player) <= 7 and game.board[pos[1]-((i+1)*player)][pos[0]] == 0:
                    result.append((pos[0], pos[1]-((i+1)*player)))
                else:
                    break
            if pos[1]-player <= 7 and pos[1]-player >= 0 and pos[0]+1 <= 7 and pos[0]+1 >= 0 and game.board[pos[1]-player][pos[0]+1]*player < 0:
                result.append((pos[0]+1, pos[1]-player))
            if pos[1]-player >= 0 and pos[1]-player <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and game.board[pos[1]-player][pos[0]-1]*player < 0:
                result.append((pos[0]-1, pos[1]-player))
            if pos[1]-player <= 7 and pos[1]-player >= 0 and pos[0]+1 <= 7 and pos[0]+1 >= 0 and (pos[0]+1, pos[1]-player) == game.en_passant:
                result.append((pos[0]+1, pos[1]-player))
            if pos[1]-player >= 0 and pos[1]-player <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (pos[0]-1, pos[1]-player) == game.en_passant:
                result.append((pos[0]-1, pos[1]-player))
            return result

if __name__ == '__main__':
    #chess_game = Chess(EPD='4kb2/rpp1p3/6p1/6Np/3Q1B2/4P2b/PPP2PPP/RN1R2K1 w - -')
    chess_game = Chess(EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -')
    #chess_game = Chess()
    while True:
        if chess_game.p_move == 1:
            print('\nWhites Turn [UPPER CASE]\n')
        else:
            print('\nBlacks Turn [LOWER CASE]\n')
        chess_game.display()
        cur = input('What piece do you want to move?\n')
        next = input('Where do you want to move the piece to?\n')
        if chess_game.move(cur, next) == False:
            if len(chess_game.log) > 0 and '+' in chess_game.log[-1]:
                print('Invalid move, you are in check')
            else:
                print('Invalid move')
        else:
            state = chess_game.is_end()
            if sum(state) > 0:
                print('\n*********************\n      GAME OVER\n*********************\n')
                chess_game.display()
                print('Game Log:\n---------\n')
                print(f'INITIAL POSITION = {chess_game.init_pos}')
                print(f'MOVES = {chess_game.log}')
                print('\nGame Result:\n------------\n')
                if state == [0, 0, 1]:
                    print('BLACK WINS\n')
                elif state == [1, 0, 0]:
                    print('WHITE WINS\n')
                else:
                    print('TIE GAME\n')
                break

            chess_game.p_move = chess_game.p_move * (-1)