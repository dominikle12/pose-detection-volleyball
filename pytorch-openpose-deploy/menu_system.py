import cv2
import numpy as np
import time
from typing import Optional, Tuple, List
from leaderboard import LeaderboardManager

class MenuSystem:
    """Handles all menu screens and user interface"""
    
    def __init__(self, canvas_width: int, canvas_height: int):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.leaderboard = LeaderboardManager()
        self.current_player = ""
        self.input_active = False
        self.input_text = ""
        self.cursor_visible = True
        self.cursor_blink_time = 0
        self.selected_menu_item = 0
        self.game_start_time = 0
        self.final_score = 0
        self.final_game_duration = 0.0
        
        # Menu states
        self.MAIN_MENU = "main_menu"
        self.LEADERBOARD = "leaderboard"
        self.NAME_INPUT = "name_input"
        self.GAME_OVER = "game_over"
        self.PLAYING = "playing"
        
        self.current_state = self.MAIN_MENU
        
    def set_game_start_time(self):
        """Set the game start time for duration tracking"""
        self.game_start_time = time.time()
        
    def get_game_duration(self) -> float:
        """Get current or final game duration in seconds"""
        if self.game_start_time == 0:
            return 0.0
        # If game is over, return the stored final duration
        if self.current_state == self.GAME_OVER:
            return self.final_game_duration
        # Otherwise return current duration
        return time.time() - self.game_start_time
    
    def draw_centered_text(self, canvas, text: str, y: int, font_scale: float = 1.0, 
                          color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2):
        """Draw text centered horizontally on the canvas"""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = (self.canvas_width - text_size[0]) // 2
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return x, y
    
    def draw_menu_item(self, canvas, text: str, y: int, is_selected: bool = False, 
                      font_scale: float = 0.8):
        """Draw a menu item with selection highlighting"""
        color = (0, 255, 255) if is_selected else (255, 255, 255)
        thickness = 3 if is_selected else 2
        
        if is_selected:
            # Draw selection background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x = (self.canvas_width - text_size[0]) // 2
            padding = 10
            cv2.rectangle(canvas, 
                         (x - padding, y - text_size[1] - padding),
                         (x + text_size[0] + padding, y + padding),
                         (50, 50, 50), -1)
            cv2.rectangle(canvas, 
                         (x - padding, y - text_size[1] - padding),
                         (x + text_size[0] + padding, y + padding),
                         color, 2)
        
        self.draw_centered_text(canvas, text, y, font_scale, color, thickness)
    
    def draw_main_menu(self, canvas):
        """Draw the main menu screen"""
        # Clear canvas with dark background
        canvas.fill(30)
        
        # Title
        self.draw_centered_text(canvas, "BALL PHYSICS DEMO", 100, 1.5, (0, 255, 255), 3)
        self.draw_centered_text(canvas, "Volleyball Edition", 140, 0.8, (255, 255, 0), 2)
        
        # Menu items
        menu_items = ["Start Game", "Leaderboard", "Quit"]
        start_y = 250
        spacing = 60
        
        for i, item in enumerate(menu_items):
            self.draw_menu_item(canvas, item, start_y + i * spacing, i == self.selected_menu_item)
        
        # Instructions
        self.draw_centered_text(canvas, "Use UP/DOWN arrows to navigate, ENTER to select", 
                               self.canvas_height - 80, 0.6, (200, 200, 200), 1)
        
        # Statistics
        total_players = self.leaderboard.get_total_players()
        if total_players > 0:
            top_scores = self.leaderboard.get_top_scores(1)
            if top_scores:
                best_score = top_scores[0]
                stats_text = f"Players: {total_players} | Best Score: {best_score['score']} by {best_score['player_name']}"
                self.draw_centered_text(canvas, stats_text, self.canvas_height - 40, 0.5, (150, 150, 150), 1)
    
    def draw_name_input(self, canvas):
        """Draw the name input screen"""
        canvas.fill(30)
        
        self.draw_centered_text(canvas, "ENTER YOUR NAME", 150, 1.2, (0, 255, 255), 3)
        
        # Input box
        box_width = 400
        box_height = 60
        box_x = (self.canvas_width - box_width) // 2
        box_y = 220
        
        # Draw input box
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 100, 100), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 255), 2)
        
        # Draw input text
        display_text = self.input_text
        if len(display_text) > 20:
            display_text = display_text[-20:]  # Show last 20 characters
            
        # Add cursor
        if self.cursor_visible:
            display_text += "|"
            
        text_x = box_x + 10
        text_y = box_y + 40
        cv2.putText(canvas, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Instructions
        self.draw_centered_text(canvas, "Type your name and press ENTER to start", 
                               350, 0.7, (200, 200, 200), 2)
        self.draw_centered_text(canvas, "ESC to go back to main menu", 
                               380, 0.6, (150, 150, 150), 1)
    
    def draw_leaderboard(self, canvas):
        """Draw the leaderboard screen"""
        canvas.fill(30)
        
        self.draw_centered_text(canvas, "LEADERBOARD", 80, 1.3, (0, 255, 255), 3)
        
        top_scores = self.leaderboard.get_top_scores(10)
        
        if not top_scores:
            self.draw_centered_text(canvas, "No scores yet! Be the first to play!", 
                                   self.canvas_height // 2, 0.8, (255, 255, 0), 2)
        else:
            # Header
            header_y = 140
            cv2.putText(canvas, "RANK", (50, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, "PLAYER", (150, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, "SCORE", (350, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, "TIME", (450, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, "DATE", (550, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw line under header
            cv2.line(canvas, (40, header_y + 10), (self.canvas_width - 40, header_y + 10), (100, 100, 100), 1)
            
            # Scores
            start_y = 170
            for i, entry in enumerate(top_scores):
                y = start_y + i * 30
                rank = i + 1
                
                # Different colors for top 3
                if rank == 1:
                    color = (0, 215, 255)  # Gold
                elif rank == 2:
                    color = (192, 192, 192)  # Silver
                elif rank == 3:
                    color = (139, 69, 19)  # Bronze
                else:
                    color = (255, 255, 255)
                
                cv2.putText(canvas, f"{rank}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(canvas, entry['player_name'][:15], (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(canvas, f"{entry['score']}", (350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(canvas, f"{entry['game_duration']}s", (450, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(canvas, entry['date'][:10], (550, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Instructions
        self.draw_centered_text(canvas, "Press ESC to return to main menu", 
                               self.canvas_height - 40, 0.6, (200, 200, 200), 1)
    
    def draw_game_over(self, canvas, final_score: int, game_duration: float):
        """Draw the game over screen"""
        canvas.fill(30)
        
        self.draw_centered_text(canvas, "GAME OVER", 120, 1.5, (255, 0, 0), 3)
        
        # Score display
        self.draw_centered_text(canvas, f"Final Score: {final_score}", 200, 1.2, (0, 255, 255), 3)
        self.draw_centered_text(canvas, f"Game Duration: {game_duration:.1f} seconds", 250, 0.8, (255, 255, 0), 2)
        
        # Check if it's a high score
        top_scores = self.leaderboard.get_top_scores(10)
        is_high_score = len(top_scores) < 10 or final_score > top_scores[-1]['score']
        
        if is_high_score and final_score > 0:
            self.draw_centered_text(canvas, "NEW HIGH SCORE!", 300, 1.0, (255, 255, 0), 2)
            if self.current_player:
                rank = self.leaderboard.get_player_rank(self.current_player)
                if rank:
                    self.draw_centered_text(canvas, f"You are rank #{rank}!", 330, 0.8, (0, 255, 0), 2)
        
        # Menu options
        menu_items = ["Play Again", "Main Menu", "View Leaderboard"]
        start_y = 400
        spacing = 50
        
        for i, item in enumerate(menu_items):
            self.draw_menu_item(canvas, item, start_y + i * spacing, i == self.selected_menu_item, 0.7)
        
        # Instructions
        self.draw_centered_text(canvas, "Use UP/DOWN arrows and ENTER to select", 
                               self.canvas_height - 40, 0.6, (200, 200, 200), 1)
    
    def update_cursor_blink(self):
        """Update cursor blinking animation"""
        if time.time() - self.cursor_blink_time > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_blink_time = time.time()
    
    def handle_key_input(self, key: int) -> Optional[str]:
        """Handle keyboard input and return action if any"""
        if self.current_state == self.MAIN_MENU:
            return self.handle_main_menu_input(key)
        elif self.current_state == self.NAME_INPUT:
            return self.handle_name_input(key)
        elif self.current_state == self.LEADERBOARD:
            return self.handle_leaderboard_input(key)
        elif self.current_state == self.GAME_OVER:
            return self.handle_game_over_input(key)
        return None
    
    def handle_main_menu_input(self, key: int) -> Optional[str]:
        """Handle main menu keyboard input"""
        if key == ord('w') or key == 82:  # W or UP arrow
            self.selected_menu_item = (self.selected_menu_item - 1) % 3
        elif key == ord('s') or key == 84:  # S or DOWN arrow
            self.selected_menu_item = (self.selected_menu_item + 1) % 3
        elif key == 13:  # ENTER
            if self.selected_menu_item == 0:  # Start Game
                self.current_state = self.NAME_INPUT
                self.input_text = ""
                return "start_name_input"
            elif self.selected_menu_item == 1:  # Leaderboard
                self.current_state = self.LEADERBOARD
                return "show_leaderboard"
            elif self.selected_menu_item == 2:  # Quit
                return "quit"
        return None
    
    def handle_name_input(self, key: int) -> Optional[str]:
        """Handle name input keyboard input"""
        if key == 27:  # ESC
            self.current_state = self.MAIN_MENU
            return "back_to_menu"
        elif key == 13:  # ENTER
            if self.input_text.strip():
                self.current_player = self.input_text.strip()[:20]
                self.current_state = self.PLAYING
                return "start_game"
        elif key == 8:  # BACKSPACE
            if self.input_text:
                self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126:  # Printable ASCII characters
            if len(self.input_text) < 20:
                self.input_text += chr(key)
        return None
    
    def handle_leaderboard_input(self, key: int) -> Optional[str]:
        """Handle leaderboard keyboard input"""
        if key == 27:  # ESC
            self.current_state = self.MAIN_MENU
            return "back_to_menu"
        return None
    
    def handle_game_over_input(self, key: int) -> Optional[str]:
        """Handle game over screen keyboard input"""
        if key == ord('w') or key == 82:  # W or UP arrow
            self.selected_menu_item = (self.selected_menu_item - 1) % 3
        elif key == ord('s') or key == 84:  # S or DOWN arrow
            self.selected_menu_item = (self.selected_menu_item + 1) % 3
        elif key == 13:  # ENTER
            if self.selected_menu_item == 0:  # Play Again
                self.current_state = self.PLAYING
                return "restart_game"
            elif self.selected_menu_item == 1:  # Main Menu
                self.current_state = self.MAIN_MENU
                return "back_to_menu"
            elif self.selected_menu_item == 2:  # View Leaderboard
                self.current_state = self.LEADERBOARD
                return "show_leaderboard"
        return None
    
    def save_score(self, score: int) -> bool:
        """Save the current player's score"""
        if not self.current_player or score <= 0:
            return False
        
        game_duration = self.get_game_duration()
        return self.leaderboard.add_score(self.current_player, score, game_duration)
    
    def show_game_over(self, final_score: int):
        """Show game over screen with final score"""
        self.final_score = final_score
        # Store the final game duration when game ends
        self.final_game_duration = time.time() - self.game_start_time if self.game_start_time > 0 else 0.0
        self.current_state = self.GAME_OVER
        self.selected_menu_item = 0
        
        # Save score if player name exists and score > 0
        if self.current_player and final_score > 0:
            self.save_score(final_score)
    
    def draw_current_screen(self, canvas):
        """Draw the current screen based on state"""
        self.update_cursor_blink()
        
        if self.current_state == self.MAIN_MENU:
            self.draw_main_menu(canvas)
        elif self.current_state == self.NAME_INPUT:
            self.draw_name_input(canvas)
        elif self.current_state == self.LEADERBOARD:
            self.draw_leaderboard(canvas)
        elif self.current_state == self.GAME_OVER:
            self.draw_game_over(canvas, self.final_score, self.get_game_duration())
    
    def is_playing(self) -> bool:
        """Check if currently in game playing state"""
        return self.current_state == self.PLAYING
    
    def is_in_menu(self) -> bool:
        """Check if currently in any menu state"""
        return self.current_state != self.PLAYING