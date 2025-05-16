import pygame
import sys
import time
import traceback
from game import Game
from ai_player import MinimaxAI
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quoridor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

class Menu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Quoridor")
        
        # Load fonts
        try:
            self.font = pygame.font.Font(None, 74)
            self.small_font = pygame.font.Font(None, 36)
        except pygame.error:
            logger.warning("Font loading error. Using system default.")
            self.font = pygame.font.SysFont('Arial', 74)
            self.small_font = pygame.font.SysFont('Arial', 36)
        
        # Create buttons
        button_width = 300
        button_height = 80
        spacing = 20
        start_y = WINDOW_HEIGHT // 2 - (2 * button_height + spacing) // 2
        self.buttons = {
            'pvp': pygame.Rect(WINDOW_WIDTH//2 - button_width//2, start_y, button_width, button_height),
            'pve': pygame.Rect(WINDOW_WIDTH//2 - button_width//2, start_y + button_height + spacing, button_width, button_height),
            'advanced': pygame.Rect(WINDOW_WIDTH//2 - button_width//2, start_y + 2*(button_height + spacing), button_width, button_height)
        }
        
        # Initialize clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # AI difficulty
        self.ai_depth = 2
        self.ai_radius = 3
        self.ai_timeout = 30
    
    def draw_button(self, rect, text, hover=False):
        color = LIGHT_GRAY if hover else GRAY
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)  # Border
        text_surface = self.small_font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def display_error(self, error_message):
        """Display an error message on screen"""
        self.screen.fill(WHITE)
        
        # Draw error title
        title = self.font.render("Error Occurred", True, RED)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 100))
        self.screen.blit(title, title_rect)
        
        # Draw error message (split into multiple lines if needed)
        lines = [error_message[i:i+50] for i in range(0, len(error_message), 50)]
        for i, line in enumerate(lines):
            msg = self.small_font.render(line, True, BLACK)
            msg_rect = msg.get_rect(center=(WINDOW_WIDTH//2, 200 + i*30))
            self.screen.blit(msg, msg_rect)
        
        # Draw continue button
        continue_button = pygame.Rect(WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT - 100, 200, 50)
        self.draw_button(continue_button, "Continue")
        
        pygame.display.flip()
        
        # Wait for user to click continue or exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if continue_button.collidepoint(event.pos):
                        waiting = False
    
    def run_game_mode(self, mode):
        """Run the selected game mode with error handling"""
        try:
            if mode == 'pvp':
                game = Game()
                game.run_pvp()
            elif mode == 'pve':
                # Create a new Minimax AI instance with a timeout
                minimax_ai = MinimaxAI(max_depth=self.ai_depth, wall_radius=self.ai_radius)
                # Ajoutez cette ligne pour définir le timeout de l'IA
                minimax_ai.timeout = self.ai_timeout
                game = Game()
                game.run_pve(minimax_ai)
            elif mode == 'advanced':
                self.show_ai_settings()
        except Exception as e:
            logger.error(f"Error in game: {str(e)}")
            logger.error(traceback.format_exc())
            self.display_error(f"Game error: {str(e)}")
    
    def show_ai_settings(self):
        """Show AI difficulty settings screen"""
        running = True
        
        # Create buttons
        button_width = 100
        button_height = 50
        spacing = 20
        start_y = WINDOW_HEIGHT // 2 - button_height // 2 - 50
        
        depth_minus = pygame.Rect(WINDOW_WIDTH//2 - 150, start_y, button_height, button_height)
        depth_plus = pygame.Rect(WINDOW_WIDTH//2 - 150 + button_width + 20, start_y, button_height, button_height)
        
        radius_minus = pygame.Rect(WINDOW_WIDTH//2 + 50, start_y, button_height, button_height)
        radius_plus = pygame.Rect(WINDOW_WIDTH//2 + 50 + button_width + 20, start_y, button_height, button_height)
        
        start_y +=30
        # Ajoutez ces lignes pour les boutons de timeout
        timeout_minus = pygame.Rect(WINDOW_WIDTH//2 - 150, start_y + 70, button_height, button_height)
        timeout_plus = pygame.Rect(WINDOW_WIDTH//2 - 150 + button_width + 20, start_y + 70, button_height, button_height)
        
        back_button = pygame.Rect(WINDOW_WIDTH//2 - 100, start_y + 170, 200, button_height)  # Déplacé plus bas
        
        while running:
            self.screen.fill(WHITE)
            
            # Draw title
            title = self.font.render("AI Settings", True, BLACK)
            title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 100))
            self.screen.blit(title, title_rect)
            
            # Draw depth settings
            depth_text = self.small_font.render(f"Search Depth: {self.ai_depth}", True, BLACK)
            self.screen.blit(depth_text, (WINDOW_WIDTH//2 - 150, start_y - 70))
            
            # Draw radius settings
            radius_text = self.small_font.render(f"Wall Radius: {self.ai_radius}", True, BLACK)
            self.screen.blit(radius_text, (WINDOW_WIDTH//2 + 50, start_y - 70))
            
            # Ajoutez ces lignes pour afficher le réglage du temps
            timeout_text = self.small_font.render(f"AI Time (s): {self.ai_timeout}", True, BLACK)
            self.screen.blit(timeout_text, (WINDOW_WIDTH//2 - 150, start_y + 30))
            
            # Draw buttons
            self.draw_button(depth_minus, "-")
            self.draw_button(depth_plus, "+")
            self.draw_button(radius_minus, "-")
            self.draw_button(radius_plus, "+")
            # Ajoutez ces lignes pour dessiner les boutons de timeout
            self.draw_button(timeout_minus, "-")
            self.draw_button(timeout_plus, "+")
            self.draw_button(back_button, "Back")
            
            # Get mouse position for hover effect
            mouse_pos = pygame.mouse.get_pos()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if depth_minus.collidepoint(event.pos) and self.ai_depth > 1:
                        self.ai_depth -= 1
                    elif depth_plus.collidepoint(event.pos) and self.ai_depth < 4:
                        self.ai_depth += 1
                    elif radius_minus.collidepoint(event.pos) and self.ai_radius > 1:
                        self.ai_radius -= 1
                    elif radius_plus.collidepoint(event.pos) and self.ai_radius < 5:
                        self.ai_radius += 1
                    # Ajoutez ces lignes pour gérer les clics sur les boutons de timeout
                    elif timeout_minus.collidepoint(event.pos) and self.ai_timeout > 1:
                        self.ai_timeout -= 1
                    elif timeout_plus.collidepoint(event.pos) and self.ai_timeout < 120:
                        self.ai_timeout += 1
                    elif back_button.collidepoint(event.pos):
                        running = False
            
            pygame.display.flip()
            self.clock.tick(60)
    
    def run(self):
        running = True
        while running:
            self.screen.fill(WHITE)
            
            # Draw title
            title = self.font.render("Quoridor", True, BLACK)
            title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 100))
            self.screen.blit(title, title_rect)
            
            # Get mouse position for hover effect
            mouse_pos = pygame.mouse.get_pos()
            
            # Draw buttons with hover effect
            button_texts = {
                'pvp': "Player vs Player", 
                'pve': "Player vs AI",
                'advanced': "AI Settings"
            }
            
            for button_name, button_rect in self.buttons.items():
                hover = button_rect.collidepoint(mouse_pos)
                self.draw_button(button_rect, button_texts[button_name], hover)
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for button_name, button_rect in self.buttons.items():
                        if button_rect.collidepoint(event.pos):
                            self.run_game_mode(button_name)
            
            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    try:
        # Check if command line arguments were provided
        if len(sys.argv) > 1 and sys.argv[1] == "--ai":
            # Start directly with AI game
            logger.info("Starting game with AI directly")
            minimax_ai = MinimaxAI(max_depth=2, wall_radius=3)
            game = Game()
            game.run_pve(minimax_ai)
            # After the game ends, show the menu
            menu = Menu()
            menu.run()
        else:
            # Start with menu
            logger.info("Starting game with menu")
            menu = Menu()
            menu.run()
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"Critical error occurred: {str(e)}")
        print("See quoridor.log for details")