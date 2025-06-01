import pygame
import sys
import json
import os
import threading
import time
import importlib
import numpy as np

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Constants
WINDOW_SIZE = (800, 600)
BOARD_SIZE = 400
CELL_SIZE = BOARD_SIZE // 5
MARGIN = (WINDOW_SIZE[0] - BOARD_SIZE) // 2
TOP_MARGIN = 100
LAYER_BUTTON_SIZE = 40

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
LIGHT_BLUE = (0, 150, 255)
YELLOW = (255, 255, 0)
DARK_GRAY = (50, 50, 50)
VERY_LIGHT_GRAY = (200, 200, 200)
VERY_LIGHT_RED = (255, 100, 100)
VERY_LIGHT_BLUE = (100, 200, 255)
GREEN = (0, 255, 0)
BACKGROUND_COLOR = (20, 20, 40)  # Dark blue background
BUTTON_COLOR = (40, 40, 80)      # Darker blue for buttons
BUTTON_HOVER = (60, 60, 100)     # Lighter blue for button hover

# Game states
MENU = 0
GAME = 1
RULES = 2
SCORES = 3
SELECT_FIRST = 4

# Load sounds
try:
    click_sound = pygame.mixer.Sound("sounds/click.wav")
    win_sound = pygame.mixer.Sound("sounds/win.wav")
    lose_sound = pygame.mixer.Sound("sounds/lose.wav")
    # Set volume for all sounds
    click_sound.set_volume(0.5)
    win_sound.set_volume(0.5)
    lose_sound.set_volume(0.5)
except Exception as e:
    print(f"Error loading sounds: {e}")
    click_sound = pygame.mixer.Sound(buffer=b'\x00' * 44100)
    win_sound = pygame.mixer.Sound(buffer=b'\x00' * 44100)
    lose_sound = pygame.mixer.Sound(buffer=b'\x00' * 44100)

class GameInterface:
    def __init__(self):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("3D Connect-4")
        self.clock = pygame.time.Clock()
        # Load 8-bit font
        self.font = pygame.font.Font("font/PressStart2P-Regular.ttf", 20)
        self.small_font = pygame.font.Font("font/PressStart2P-Regular.ttf", 14)
        self.state = MENU
        self.current_layer = 0
        self.scores = self.load_scores()
        self.player_turn = True
        self.game_over = False
        self.message = None
        self.error_message = None
        self.error_timer = 0
        self.ai_thinking = False
        self.last_player_move = None
        self.last_ai_move = None
        self.ai_move_thread = None
        self.ai_move_result = None
        self.main_module = None
        self.load_main_module()
        self.winning_combination = None
        self.winning_player = None
        self.button_hover = None  # Track which button is being hovered
        
    def load_main_module(self):
        self.main_module = importlib.import_module('minimax')
        importlib.reload(self.main_module)
        
    def load_scores(self):
        if os.path.exists('scores.json'):
            with open('scores.json', 'r') as f:
                return json.load(f)
        return {"player": 0, "ai": 0}
    
    def save_scores(self):
        with open('scores.json', 'w') as f:
            json.dump(self.scores, f)
    
    def draw_menu(self):
        self.screen.fill(BACKGROUND_COLOR)
        title = self.font.render("3D Connect-4", True, WHITE)
        start = self.font.render("Start Game", True, WHITE)
        rules = self.font.render("Rules", True, WHITE)
        scores = self.font.render("Scores", True, WHITE)
        quit_text = self.font.render("Quit", True, WHITE)
        
        # Draw buttons with hover effect
        buttons = [
            (300, 200, 200, 50, start, "start"),
            (300, 270, 200, 50, rules, "rules"),
            (300, 340, 200, 50, scores, "scores"),
            (300, 410, 200, 50, quit_text, "quit")
        ]
        
        mouse_pos = pygame.mouse.get_pos()
        self.button_hover = None
        
        for x, y, w, h, text, name in buttons:
            color = BUTTON_HOVER if (x <= mouse_pos[0] <= x + w and y <= mouse_pos[1] <= y + h) else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, (x, y, w, h))
            pygame.draw.rect(self.screen, WHITE, (x, y, w, h), 2)  # White border
            self.screen.blit(text, (WINDOW_SIZE[0]//2 - text.get_width()//2, y + h//2 - text.get_height()//2))
            if (x <= mouse_pos[0] <= x + w and y <= mouse_pos[1] <= y + h):
                self.button_hover = name
        
        self.screen.blit(title, (WINDOW_SIZE[0]//2 - title.get_width()//2, 100))
    
    def draw_rules(self):
        self.screen.fill(BACKGROUND_COLOR)
        title = self.font.render("Game Rules", True, WHITE)
        rules = [
            "1. The game is played on a 5x5x5 board",
            "2. Players take turns placing pieces",
            "3. Pieces fall to the bottom due to gravity",
            "4. Connect 4 pieces in any direction to win",
            "5. Directions include: straight lines and diagonals",
            "6. First player to connect 4 wins!",
            "Controls:",
            "- Click to place a piece",
            "- Use layer buttons on the right to switch layers",
            "Press ESC to return to menu"
        ]
        
        self.screen.blit(title, (WINDOW_SIZE[0]//2 - title.get_width()//2, 50))
        for i, rule in enumerate(rules):
            text = self.small_font.render(rule, True, WHITE)
            self.screen.blit(text, (50, 120 + i * 30))
    
    def draw_scores(self):
        self.screen.fill(BACKGROUND_COLOR)
        title = self.font.render("Scores", True, WHITE)
        player_score = self.font.render(f"Player: {self.scores['player']}", True, WHITE)
        ai_score = self.font.render(f"AI: {self.scores['ai']}", True, WHITE)
        back = self.font.render("Press ESC to return to menu", True, WHITE)
        
        self.screen.blit(title, (WINDOW_SIZE[0]//2 - title.get_width()//2, 100))
        self.screen.blit(player_score, (WINDOW_SIZE[0]//2 - player_score.get_width()//2, 200))
        self.screen.blit(ai_score, (WINDOW_SIZE[0]//2 - ai_score.get_width()//2, 250))
        self.screen.blit(back, (WINDOW_SIZE[0]//2 - back.get_width()//2, 400))

    def draw_select_first(self):
        self.screen.fill(BACKGROUND_COLOR)
        title = self.font.render("Select First Player", True, WHITE)

        # Render first player selection with arrows
        left_arrow = self.font.render(">", True, WHITE)
        right_arrow = self.font.render("<", True, WHITE)
        first_label = self.font.render("Player First" if self.player_turn else "AI First", True, WHITE)

        # Position for label and arrows
        center_x = WINDOW_SIZE[0] // 2
        label_y = 300
        spacing = 50

        label_width = first_label.get_width()
        arrow_y = label_y

        # Arrows' positions
        left_x = center_x - label_width // 2 - spacing
        label_x = center_x - label_width // 2
        right_x = center_x + label_width // 2 + spacing - right_arrow.get_width()

        # Draw left arrow (button)
        left_rect = pygame.Rect(left_x, arrow_y, left_arrow.get_width(), left_arrow.get_height())
        left_color = BUTTON_HOVER if left_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.screen, left_color, left_rect)
        self.screen.blit(left_arrow, (left_x, arrow_y))

        # Draw label (not a button)
        self.screen.blit(first_label, (label_x, label_y))

        # Draw right arrow (button)
        right_rect = pygame.Rect(right_x, arrow_y, right_arrow.get_width(), right_arrow.get_height())
        right_color = BUTTON_HOVER if right_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.screen, right_color, right_rect)
        self.screen.blit(right_arrow, (right_x, arrow_y))

        # Start button (without border)
        start_button = self.font.render("Start", True, WHITE)
        start_rect = pygame.Rect(
            WINDOW_SIZE[0] // 2 - start_button.get_width() // 2, 400,
            start_button.get_width(), start_button.get_height()
        )
        start_color = BUTTON_HOVER if start_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.screen, start_color, start_rect)
        self.screen.blit(start_button, (WINDOW_SIZE[0] // 2 - start_button.get_width() // 2, 400))

        # Draw title
        self.screen.blit(title, (WINDOW_SIZE[0] // 2 - title.get_width() // 2, 200))

        # Optionally: return rects for click handling
        return left_rect, right_rect, start_rect

    def draw_layer_buttons(self):
        button_x = WINDOW_SIZE[0] - LAYER_BUTTON_SIZE - 20
        button_y = TOP_MARGIN
        
        # Draw up button
        pygame.draw.polygon(self.screen, DARK_GRAY if self.current_layer < 4 else GRAY,
                          [(button_x, button_y + LAYER_BUTTON_SIZE),
                           (button_x + LAYER_BUTTON_SIZE//2, button_y),
                           (button_x + LAYER_BUTTON_SIZE, button_y + LAYER_BUTTON_SIZE)])
        
        # Draw current layer number
        layer_text = self.font.render(str(self.current_layer + 1), True, WHITE)
        self.screen.blit(layer_text, (button_x + LAYER_BUTTON_SIZE//2 - layer_text.get_width()//2,
                                    button_y + LAYER_BUTTON_SIZE + 10))
        
        # Draw down button
        pygame.draw.polygon(self.screen, DARK_GRAY if self.current_layer > 0 else GRAY,
                          [(button_x, button_y + LAYER_BUTTON_SIZE * 2),
                           (button_x + LAYER_BUTTON_SIZE//2, button_y + LAYER_BUTTON_SIZE * 3),
                           (button_x + LAYER_BUTTON_SIZE, button_y + LAYER_BUTTON_SIZE * 2)])
    
    def draw_board(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw layer selector
        layer_text = self.font.render(f"Layer: {self.current_layer + 1}", True, WHITE)
        self.screen.blit(layer_text, (50, 50))
        
        # Draw board
        for x in range(5):
            for y in range(5):
                # Draw cell background
                cell_rect = pygame.Rect(MARGIN + x * CELL_SIZE, 
                                      TOP_MARGIN + y * CELL_SIZE, 
                                      CELL_SIZE, CELL_SIZE)
                
                # If this cell is part of winning combination, highlight it
                if self.winning_combination and (x+1, y+1, self.current_layer+1) in self.winning_combination:
                    pygame.draw.rect(self.screen, GREEN, cell_rect)
                
                pygame.draw.rect(self.screen, WHITE, cell_rect, 2)  # White grid lines
                
                # Draw piece
                if self.main_module.board[x, y, self.current_layer] == self.main_module.PLAYER:
                    color = VERY_LIGHT_BLUE if self.ai_thinking else LIGHT_BLUE
                    pygame.draw.circle(self.screen, color,
                                    (MARGIN + x * CELL_SIZE + CELL_SIZE//2,
                                     TOP_MARGIN + y * CELL_SIZE + CELL_SIZE//2),
                                    CELL_SIZE//2 - 5)
                elif self.main_module.board[x, y, self.current_layer] == self.main_module.AI:
                    color = VERY_LIGHT_RED if self.ai_thinking else RED
                    pygame.draw.circle(self.screen, color,
                                    (MARGIN + x * CELL_SIZE + CELL_SIZE//2,
                                     TOP_MARGIN + y * CELL_SIZE + CELL_SIZE//2),
                                    CELL_SIZE//2 - 5)
                
                # Highlight last moves
                if self.last_player_move and (x+1, y+1, self.current_layer+1) == self.last_player_move:
                    pygame.draw.circle(self.screen, YELLOW,
                                    (MARGIN + x * CELL_SIZE + CELL_SIZE//2,
                                     TOP_MARGIN + y * CELL_SIZE + CELL_SIZE//2),
                                    CELL_SIZE//2 - 2, 2)
                if self.last_ai_move and (x+1, y+1, self.current_layer+1) == self.last_ai_move:
                    pygame.draw.circle(self.screen, YELLOW,
                                    (MARGIN + x * CELL_SIZE + CELL_SIZE//2,
                                     TOP_MARGIN + y * CELL_SIZE + CELL_SIZE//2),
                                    CELL_SIZE//2 - 2, 2)
        
        # Draw layer buttons with hover effect
        button_x = WINDOW_SIZE[0] - LAYER_BUTTON_SIZE - 20
        button_y = TOP_MARGIN
        mouse_pos = pygame.mouse.get_pos()
        
        # Draw up button
        up_color = BUTTON_HOVER if (button_x <= mouse_pos[0] <= button_x + LAYER_BUTTON_SIZE and 
                                  button_y <= mouse_pos[1] <= button_y + LAYER_BUTTON_SIZE) else BUTTON_COLOR
        pygame.draw.polygon(self.screen, up_color if self.current_layer < 4 else DARK_GRAY,
                          [(button_x, button_y + LAYER_BUTTON_SIZE),
                           (button_x + LAYER_BUTTON_SIZE//2, button_y),
                           (button_x + LAYER_BUTTON_SIZE, button_y + LAYER_BUTTON_SIZE)])
        pygame.draw.polygon(self.screen, WHITE,
                          [(button_x, button_y + LAYER_BUTTON_SIZE),
                           (button_x + LAYER_BUTTON_SIZE//2, button_y),
                           (button_x + LAYER_BUTTON_SIZE, button_y + LAYER_BUTTON_SIZE)], 2)
        
        # Draw current layer number
        layer_text = self.font.render(str(self.current_layer + 1), True, WHITE)
        self.screen.blit(layer_text, (button_x + LAYER_BUTTON_SIZE//2 - layer_text.get_width()//2,
                                    button_y + LAYER_BUTTON_SIZE + 10))
        
        # Draw down button
        down_color = BUTTON_HOVER if (button_x <= mouse_pos[0] <= button_x + LAYER_BUTTON_SIZE and 
                                    button_y + LAYER_BUTTON_SIZE * 2 <= mouse_pos[1] <= button_y + LAYER_BUTTON_SIZE * 3) else BUTTON_COLOR
        pygame.draw.polygon(self.screen, down_color if self.current_layer > 0 else DARK_GRAY,
                          [(button_x, button_y + LAYER_BUTTON_SIZE * 2),
                           (button_x + LAYER_BUTTON_SIZE//2, button_y + LAYER_BUTTON_SIZE * 3),
                           (button_x + LAYER_BUTTON_SIZE, button_y + LAYER_BUTTON_SIZE * 2)])
        pygame.draw.polygon(self.screen, WHITE,
                          [(button_x, button_y + LAYER_BUTTON_SIZE * 2),
                           (button_x + LAYER_BUTTON_SIZE//2, button_y + LAYER_BUTTON_SIZE * 3),
                           (button_x + LAYER_BUTTON_SIZE, button_y + LAYER_BUTTON_SIZE * 2)], 2)
        
        # Draw game over message
        if self.message:
            self.screen.blit(self.message, (WINDOW_SIZE[0]//2 - self.message.get_width()//2, 50))
            
            # If game is over and there's a winning combination, show navigation hint
            if self.game_over and self.winning_combination:
                hint = self.small_font.render("Use layer buttons to view winning combination", True, WHITE)
                self.screen.blit(hint, (WINDOW_SIZE[0]//2 - hint.get_width()//2, 90))
        
        # Draw error message
        if self.error_message and self.error_timer > 0:
            self.screen.blit(self.error_message, (WINDOW_SIZE[0]//2 - self.error_message.get_width()//2, 50))
            self.error_timer -= 1
    
    def show_error(self, message):
        self.error_message = self.font.render(message, True, RED)
        self.error_timer = 60  # Show for 1 second at 60 FPS
        lose_sound.play()  # Changed from error_sound to lose_sound
    
    def handle_menu_click(self, pos):
        x, y = pos
        if 300 <= x <= 500:
            if 200 <= y <= 250:
                click_sound.play()
                self.state = SELECT_FIRST
            elif 270 <= y <= 320:
                click_sound.play()
                self.state = RULES
            elif 340 <= y <= 390:
                click_sound.play()
                self.state = SCORES
            elif 410 <= y <= 460:
                click_sound.play()
                pygame.quit()
                sys.exit()


    def handle_select_first_click(self, pos):
        x, y = pos
        first_button = self.font.render("Player First" if self.player_turn else "AI First", True, WHITE)
        start_button = self.font.render("Start", True, WHITE)

        toggle_button = pygame.Rect(WINDOW_SIZE[0]//2 - first_button.get_width()//2, 300,
                                  first_button.get_width(), first_button.get_height())
        start_rect = pygame.Rect(WINDOW_SIZE[0]//2 - start_button.get_width()//2, 400,
                               start_button.get_width(), start_button.get_height())

        if toggle_button.collidepoint(pos):
            click_sound.play()
            self.player_turn = not self.player_turn
        elif start_rect.collidepoint(pos):
            click_sound.play()
            self.state = GAME
            if not self.player_turn:
                self.ai_thinking = True
                self.ai_move_thread = threading.Thread(target=self.make_ai_move)
                self.ai_move_thread.start()
    
    def handle_game_click(self, pos):
        x, y = pos
        button_x = WINDOW_SIZE[0] - LAYER_BUTTON_SIZE - 20
        button_y = TOP_MARGIN
        
        # Check layer button clicks - allow even after game over
        if button_x <= x <= button_x + LAYER_BUTTON_SIZE:
            if button_y <= y <= button_y + LAYER_BUTTON_SIZE and self.current_layer < 4:
                click_sound.play()
                self.current_layer += 1
            elif button_y + LAYER_BUTTON_SIZE * 2 <= y <= button_y + LAYER_BUTTON_SIZE * 3 and self.current_layer > 0:
                click_sound.play()
                self.current_layer -= 1
        
        # Only process board clicks if game is not over and AI is not thinking
        if not self.game_over and not self.ai_thinking:
            # Check board clicks
            if (MARGIN <= x <= MARGIN + BOARD_SIZE and 
                TOP_MARGIN <= y <= TOP_MARGIN + BOARD_SIZE):
                board_x = (x - MARGIN) // CELL_SIZE + 1
                board_y = (y - TOP_MARGIN) // CELL_SIZE + 1
                
                # Check if the move is valid (including the first move)
                if self.main_module.valid_move(board_x, board_y, self.current_layer + 1):
                    click_sound.play()
                    self.main_module.make_move(board_x, board_y, self.current_layer + 1, self.main_module.PLAYER)
                    self.last_player_move = (board_x, board_y, self.current_layer + 1)
                    
                    if self.main_module.check_win(self.main_module.PLAYER):
                        win_sound.play()
                        self.scores['player'] += 1
                        self.save_scores()
                        self.message = self.font.render("Player Wins!", True, RED)
                        self.game_over = True
                        self.winning_combination = self.main_module.get_winning_combination()
                        self.winning_player = self.main_module.PLAYER
                    elif self.main_module.board_full():
                        self.message = self.font.render("Draw!", True, WHITE)
                        self.game_over = True
                    else:
                        self.ai_thinking = True
                        # Start AI move in a separate thread
                        self.ai_move_thread = threading.Thread(target=self.make_ai_move)
                        self.ai_move_thread.start()
                else:
                    self.show_error("Invalid move! Check gravity rule.")
    
    def make_ai_move(self):
        ai_x, ai_y, ai_z = self.main_module.ai_move()
        self.ai_move_result = (ai_x, ai_y, ai_z)
    
    def reset_game(self):
        self.load_main_module()  # Reload main module
        self.current_layer = 0
        self.game_over = False
        self.message = None
        self.error_message = None
        self.error_timer = 0
        self.ai_thinking = False
        self.last_player_move = None
        self.last_ai_move = None
        self.player_turn = True  # Reset to default player first
        self.ai_move_thread = None
        self.ai_move_result = None
        self.winning_combination = None
        self.winning_player = None

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == GAME:
                            self.reset_game()
                        self.state = MENU
                        # Force redraw of menu
                        self.screen.fill(BACKGROUND_COLOR)
                        pygame.display.flip()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == MENU:
                        self.handle_menu_click(event.pos)
                    elif self.state == SELECT_FIRST:
                        self.handle_select_first_click(event.pos)
                    elif self.state == GAME:
                        self.handle_game_click(event.pos)

            # Check if AI move is ready
            if self.ai_thinking and self.ai_move_thread and not self.ai_move_thread.is_alive():
                if self.ai_move_result:
                    ai_x, ai_y, ai_z = self.ai_move_result
                    self.main_module.make_move(ai_x, ai_y, ai_z, self.main_module.AI)
                    self.last_ai_move = (ai_x, ai_y, ai_z)
                    self.current_layer = ai_z - 1  # Switch to AI's layer
                    self.ai_thinking = False
                    self.ai_move_result = None
                    self.ai_move_thread = None

                    if self.main_module.check_win(self.main_module.AI):
                        win_sound.play()
                        self.scores['ai'] += 1
                        self.save_scores()
                        self.message = self.font.render("AI Wins!", True, BLUE)
                        self.game_over = True
                        self.winning_combination = self.main_module.get_winning_combination()
                        self.winning_player = self.main_module.AI
                    elif self.main_module.board_full():
                        self.message = self.font.render("Draw!", True, WHITE)
                        self.game_over = True

            if self.state == MENU:
                self.draw_menu()
            elif self.state == RULES:
                self.draw_rules()
            elif self.state == SCORES:
                self.draw_scores()
            elif self.state == SELECT_FIRST:
                self.draw_select_first()
            elif self.state == GAME:
                self.draw_board()

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    game = GameInterface()
    game.run() 