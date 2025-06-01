import numpy as np
import time
from numba import jit

# Board and constants
board = np.zeros((5, 5, 5), dtype=np.int32)  # 0: empty, 1: player, -1: AI
PLAYER = 1
AI = -1
directions = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],  # Axes
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],  # xy diagonals
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],  # xz diagonals
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],  # yz diagonals
    [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, 1, 1]  # 3D diagonals
], dtype=np.int32)
transposition_table = {}  # Transposition table for caching
MAX_CACHE_SIZE = 1000000  # Reduced cache size for efficiency
EARLY_DEPTH_LIMIT = 4  # Depth for early game
LATE_GAME_THRESHOLD = 31  # >50% board filled
TIME_LIMIT = 2.0  # Max seconds per AI move

def print_board():
    """Prints the board layer by layer (z=1..5)"""
    for z in range(5):
        print(f"z={z + 1}:")
        for x in range(5):
            row = ['R' if board[x, y, z] == PLAYER else 'B' if board[x, y, z] == AI else '.' for y in range(5)]
            print(' '.join(row))
        print()

def valid_move(x, y, z):
    """Checks if a move at (x, y, z) is valid"""
    x, y, z = x - 1, y - 1, z - 1
    if not (0 <= x < 5 and 0 <= y < 5 and 0 <= z < 5):
        return False
    if board[x, y, z] != 0:
        return False
    if z == 0 or board[x, y, z - 1] != 0:  # Gravity rule
        return True
    return False

def make_move(x, y, z, player):
    """Places a piece for the player at (x, y, z)"""
    x, y, z = x - 1, y - 1, z - 1
    board[x, y, z] = player

@jit(nopython=True)
def check_win_numba(board, player, last_move=None):
    """Checks if the player has won (4 in a row)"""
    if last_move is not None:
        x, y, z = last_move[0] - 1, last_move[1] - 1, last_move[2] - 1
        if board[x, y, z] != player:
            return False
        for d in range(directions.shape[0]):
            dx, dy, dz = directions[d]
            count = 1
            for step in range(1, 4):
                nx, ny, nz = x + dx * step, y + dy * step, z + dz * step
                if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5 and board[nx, ny, nz] == player:
                    count += 1
                else:
                    break
            for step in range(1, 4):
                nx, ny, nz = x - dx * step, y - dy * step, z - dz * step
                if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5 and board[nx, ny, nz] == player:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False
    else:
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if board[x, y, z] != player:
                        continue
                    for d in range(directions.shape[0]):
                        dx, dy, dz = directions[d]
                        count = 1
                        for step in range(1, 4):
                            nx, ny, nz = x + dx * step, y + dy * step, z + dz * step
                            if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5 and board[nx, ny, nz] == player:
                                count += 1
                            else:
                                break
                        if count >= 4:
                            return True
        return False

def check_win(player, last_move=None):
    """Wrapper for check_win_numba"""
    if last_move:
        return check_win_numba(board, player, np.array(last_move, dtype=np.int32))
    return check_win_numba(board, player)

def board_full():
    """Checks if the board is full"""
    return not np.any(board == 0)

def get_valid_moves():
    """Returns a list of valid moves (x, y, z)"""
    moves = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if valid_move(x + 1, y + 1, z + 1):
                    moves.append((x + 1, y + 1, z + 1))
    return moves

@jit(nopython=True)
def evaluate_position_numba(board, move_count):
    """Evaluates the board position"""
    score = 0
    moves_to_win = 100
    late_game = move_count > LATE_GAME_THRESHOLD
    triple_weight = 5000 if late_game else 3000
    triple_open_weight = 10000 if late_game else 8000
    double_open_weight = 500

    for x in range(5):
        for y in range(5):
            for z in range(5):
                if board[x, y, z] == 0:
                    continue
                player = board[x, y, z]
                for d in range(directions.shape[0]):
                    dx, dy, dz = directions[d]
                    count = 1
                    open_ends = 0
                    for step in range(1, 4):
                        nx, ny, nz = x + dx * step, y + dy * step, z + dz * step
                        if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5:
                            if board[nx, ny, nz] == player:
                                count += 1
                            elif board[nx, ny, nz] == 0:
                                open_ends += 1
                                break
                            else:
                                break
                    for step in range(1, 4):
                        nx, ny, nz = x - dx * step, y - dy * step, z - dz * step
                        if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5:
                            if board[nx, ny, nz] == player:
                                count += 1
                            elif board[nx, ny, nz] == 0:
                                open_ends += 1
                                break
                            else:
                                break
                    if count >= 4:
                        score += 100000 * player
                        moves_to_win = 0 if player == AI else -1
                    elif count == 3 and open_ends >= 1:
                        score += (triple_open_weight if open_ends >= 2 else triple_weight) * player
                        moves_to_win = min(moves_to_win, 1 if player == AI else -1)
                    elif count == 2 and open_ends >= 2:
                        score += double_open_weight * player
                        moves_to_win = min(moves_to_win, 2 if player == AI else -2)

    # Center and edge bonuses
    center = board[1:4, 1:4, 1:4]
    score += 10 * np.sum(center * center)
    if board[2, 2, 2] != 0:
        score += 50 * board[2, 2, 2]
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if board[x, y, z] != 0 and (x in (0, 4) or y in (0, 4) or z in (0, 4)):
                    score += 5 * board[x, y, z]
    return score, moves_to_win

def evaluate_position(move_count):
    """Wrapper for evaluate_position_numba"""
    return evaluate_position_numba(board, move_count)

@jit(nopython=True)
def check_threats_numba(board, player, valid_moves, move_count):
    """Checks for moves creating triples with open ends"""
    threats = np.zeros((len(valid_moves), 4), dtype=np.int32)  # [x, y, z, score]
    threat_count = 0
    min_score = 3000

    for i in range(len(valid_moves)):
        x, y, z = valid_moves[i]
        board[x - 1, y - 1, z - 1] = player
        score = 0
        for d in range(directions.shape[0]):
            dx, dy, dz = directions[d]
            count = 1
            open_ends = 0
            for step in range(1, 4):
                nx, ny, nz = x - 1 + dx * step, y - 1 + dy * step, z - 1 + dz * step
                if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5:
                    if board[nx, ny, nz] == player:
                        count += 1
                    elif board[nx, ny, nz] == 0:
                        open_ends += 1
                        break
                    else:
                        break
            for step in range(1, 4):
                nx, ny, nz = x - 1 - dx * step, y - 1 - dy * step, z - 1 - dz * step
                if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5:
                    if board[nx, ny, nz] == player:
                        count += 1
                    elif board[nx, ny, nz] == 0:
                        open_ends += 1
                        break
                    else:
                        break
            if count == 3 and open_ends >= 1:
                score += 10000 if open_ends >= 2 else 5000
        board[x - 1, y - 1, z - 1] = 0
        if score >= min_score:
            threats[threat_count] = [x, y, z, score]
            threat_count += 1
    return threats[:threat_count]

def check_threats(player, move_count):
    """Wrapper for check_threats_numba"""
    moves = get_valid_moves()
    valid_moves = np.array(moves, dtype=np.int32)
    threats_array = check_threats_numba(board, player, valid_moves, move_count)
    return [(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in threats_array]

def minimax(depth, alpha, beta, maximizing, last_move=None, start_time=None, move_count=0):
    """Minimax with alpha-beta pruning and iterative deepening"""
    if start_time and time.time() - start_time > TIME_LIMIT:
        return None, 100

    board_hash = board.tobytes()
    cache_key = (board_hash, depth, maximizing)
    if cache_key in transposition_table:
        return transposition_table[cache_key]

    if last_move and check_win(PLAYER, last_move):
        return -100000, -1
    if last_move and check_win(AI, last_move):
        return 100000, 0
    if board_full():
        return 0, 0
    if depth == 0:
        score, moves_to_win = evaluate_position(move_count)
        return score, moves_to_win

    moves = get_valid_moves()
    move_scores = []
    for x, y, z in moves:
        board[x - 1, y - 1, z - 1] = AI if maximizing else PLAYER
        score, _ = evaluate_position(move_count + 1)
        board[x - 1, y - 1, z - 1] = 0
        move_scores.append((score, (x, y, z)))
    move_scores.sort(key=lambda x: -x[0] if maximizing else x[0])
    moves = [move for _, move in move_scores]

    if maximizing:
        max_eval = -float('inf')
        best_moves_to_win = 100
        for x, y, z in moves:
            board[x - 1, y - 1, z - 1] = AI
            eval, moves_to_win = minimax(depth - 1, alpha, beta, False, (x, y, z), start_time, move_count + 1)
            board[x - 1, y - 1, z - 1] = 0
            if eval is None:
                return None, 100
            if eval > max_eval or (eval == max_eval and moves_to_win < best_moves_to_win):
                max_eval = eval
                best_moves_to_win = moves_to_win + 1
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[cache_key] = (max_eval, best_moves_to_win)
        if len(transposition_table) > MAX_CACHE_SIZE:
            transposition_table.clear()
        return max_eval, best_moves_to_win
    else:
        min_eval = float('inf')
        best_moves_to_win = 100
        for x, y, z in moves:
            board[x - 1, y - 1, z - 1] = PLAYER
            eval, moves_to_win = minimax(depth - 1, alpha, beta, True, (x, y, z), start_time, move_count + 1)
            board[x - 1, y - 1, z - 1] = 0
            if eval is None:
                return None, 100
            if eval < min_eval or (eval == min_eval and moves_to_win < best_moves_to_win):
                min_eval = eval
                best_moves_to_win = moves_to_win + 1
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[cache_key] = (min_eval, best_moves_to_win)
        if len(transposition_table) > MAX_CACHE_SIZE:
            transposition_table.clear()
        return min_eval, best_moves_to_win

def ai_move():
    """Chooses the best move for AI"""
    if len(transposition_table) > MAX_CACHE_SIZE:
        transposition_table.clear()

    moves = get_valid_moves()
    move_count = np.sum(board != 0)

    # First move: take center
    if move_count == 0:
        print("AI takes center for first move")
        return 3, 3, 1

    # # Early game: prioritize edge-adjacent positions if center is taken
    # if move_count < 5:
    #     priority_moves = [(2, 1, 1), (2, 5, 1), (4, 1, 1), (4, 5, 1)]
    #     for move in priority_moves:
    #         if move in moves:
    #             print(f"AI prioritizes edge-adjacent move {move}")
    #             return move[0], move[1], move[2]

    # Immediate win for AI
    for x, y, z in moves:
        board[x - 1, y - 1, z - 1] = AI
        if check_win(AI, (x, y, z)):
            board[x - 1, y - 1, z - 1] = 0
            print(f"AI wins with move ({x}, {y}, {z})")
            return x, y, z
        board[x - 1, y - 1, z - 1] = 0

    # Block player's immediate win
    for x, y, z in moves:
        board[x - 1, y - 1, z - 1] = PLAYER
        if check_win(PLAYER, (x, y, z)):
            board[x - 1, y - 1, z - 1] = 0
            print(f"AI blocks player's win at ({x}, {y}, {z})")
            return x, y, z
        board[x - 1, y - 1, z - 1] = 0

    # Create or block triple threats
    ai_threats = check_threats(AI, move_count)
    if ai_threats:
        ai_threats.sort(key=lambda x: (-x[3], x[2]))  # Highest score, lowest z
        print(f"AI creates threat at {ai_threats[0][:3]}")
        return ai_threats[0][0], ai_threats[0][1], ai_threats[0][2]

    player_threats = check_threats(PLAYER, move_count)
    if player_threats:
        player_threats.sort(key=lambda x: (-x[3], x[2]))
        print(f"AI blocks player's threat at {player_threats[0][:3]}")
        return player_threats[0][0], player_threats[0][1], player_threats[0][2]

    # Iterative deepening with minimax
    start_time = time.time()
    best_score = -float('inf')
    best_move = None
    best_moves_to_win = 100
    depth = 1
    max_depth = EARLY_DEPTH_LIMIT if move_count < LATE_GAME_THRESHOLD else 6
    while depth <= max_depth and time.time() - start_time < TIME_LIMIT:
        current_best_score = -float('inf')
        current_best_move = None
        current_best_moves_to_win = 100
        moves.sort(key=lambda m: (evaluate_position(move_count)[0], m[2]))
        for x, y, z in moves:
            board[x - 1, y - 1, z - 1] = AI
            score, moves_to_win = minimax(depth, -float('inf'), float('inf'), False, (x, y, z), start_time, move_count + 1)
            board[x - 1, y - 1, z - 1] = 0
            if score is None:
                break
            if score > current_best_score or (score == current_best_score and moves_to_win < current_best_moves_to_win):
                current_best_score = score
                current_best_move = (x, y, z)
                current_best_moves_to_win = moves_to_win
        if score is not None:
            best_score = current_best_score
            best_move = current_best_move
            best_moves_to_win = current_best_moves_to_win
        depth += 1

    # Fallback: take center if available
    if not best_move and (3, 3, 3) in moves:
        print("AI takes center (3, 3, 3)")
        return 3, 3, 3

    print(f"AI chooses move {best_move} with minimax")
    return best_move if best_move else moves[0]

def get_winning_combination():
    """Returns the coordinates of the winning combination if there is one."""
    # Check all possible winning combinations
    for x in range(1, 6):
        for y in range(1, 6):
            for z in range(1, 6):
                # Check horizontal
                if x <= 2:
                    if all(board[x+i-1, y-1, z-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x+i, y, z) for i in range(4)]
                # Check vertical
                if y <= 2:
                    if all(board[x-1, y+i-1, z-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x, y+i, z) for i in range(4)]
                # Check depth
                if z <= 2:
                    if all(board[x-1, y-1, z+i-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x, y, z+i) for i in range(4)]
                # Check diagonal in xy plane
                if x <= 2 and y <= 2:
                    if all(board[x+i-1, y+i-1, z-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x+i, y+i, z) for i in range(4)]
                    if all(board[x+i-1, y+3-i-1, z-1] == board[x-1, y+3-1, z-1] != 0 for i in range(4)):
                        return [(x+i, y+3-i, z) for i in range(4)]
                # Check diagonal in xz plane
                if x <= 2 and z <= 2:
                    if all(board[x+i-1, y-1, z+i-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x+i, y, z+i) for i in range(4)]
                    if all(board[x+i-1, y-1, z+3-i-1] == board[x-1, y-1, z+3-1] != 0 for i in range(4)):
                        return [(x+i, y, z+3-i) for i in range(4)]
                # Check diagonal in yz plane
                if y <= 2 and z <= 2:
                    if all(board[x-1, y+i-1, z+i-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x, y+i, z+i) for i in range(4)]
                    if all(board[x-1, y+i-1, z+3-i-1] == board[x-1, y-1, z+3-1] != 0 for i in range(4)):
                        return [(x, y+i, z+3-i) for i in range(4)]
                # Check 3D diagonals
                if x <= 2 and y <= 2 and z <= 2:
                    if all(board[x+i-1, y+i-1, z+i-1] == board[x-1, y-1, z-1] != 0 for i in range(4)):
                        return [(x+i, y+i, z+i) for i in range(4)]
                    if all(board[x+i-1, y+i-1, z+3-i-1] == board[x-1, y-1, z+3-1] != 0 for i in range(4)):
                        return [(x+i, y+i, z+3-i) for i in range(4)]
                    if all(board[x+i-1, y+3-i-1, z+i-1] == board[x-1, y+3-1, z-1] != 0 for i in range(4)):
                        return [(x+i, y+3-i, z+i) for i in range(4)]
                    if all(board[x+i-1, y+3-i-1, z+3-i-1] == board[x-1, y+3-1, z+3-1] != 0 for i in range(4)):
                        return [(x+i, y+3-i, z+3-i) for i in range(4)]
    return None

def main():
    print("Welcome to 3D Connect-4 (5x5x5)!")
    first = input("Who goes first? (player/ai): ").lower()
    player_turn = first == 'player'

    while True:
        print_board()
        if player_turn:
            while True:
                try:
                    x, y, z = map(int, input("Your move (x y z, e.g., 2 2 1): ").split())
                    if valid_move(x, y, z):
                        make_move(x, y, z, PLAYER)
                        break
                    else:
                        print("Invalid move! Check coordinates and gravity rule.")
                except ValueError:
                    print("Enter three numbers separated by spaces!")
            if check_win(PLAYER):
                print_board()
                print("You win!")
                break
            if board_full():
                print_board()
                print("Draw!")
                break
            player_turn = False
        else:
            print("AI's turn...")
            start = time.perf_counter()
            x, y, z = ai_move()
            print(f"Move time: {time.perf_counter() - start:.3f} sec")
            make_move(x, y, z, AI)
            print(f"AI moved: ({x}, {y}, {z})")
            if check_win(AI):
                print_board()
                print("AI wins!")
                break
            if board_full():
                print_board()
                print("Draw!")
                break
            player_turn = True

if __name__ == "__main__":
    main()