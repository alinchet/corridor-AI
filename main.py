import pygame
import sys
from game import Game
from ai_player import SimpleAI, SuperAI

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)

class Menu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Corridor Game")
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)
        
        # Create buttons
        button_width = 300
        button_height = 80
        spacing = 20
        start_y = WINDOW_HEIGHT // 2 - (3 * button_height + 2 * spacing) // 2
        
        self.buttons = {
            'pvp': pygame.Rect(WINDOW_WIDTH//2 - button_width//2, start_y, button_width, button_height),
            'pve_simple': pygame.Rect(WINDOW_WIDTH//2 - button_width//2, start_y + button_height + spacing, button_width, button_height),
            'pve_super': pygame.Rect(WINDOW_WIDTH//2 - button_width//2, start_y + 2 * (button_height + spacing), button_width, button_height)
        }

    def draw_button(self, rect, text, color):
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        
        text_surface = self.small_font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def run(self):
        while True:
            self.screen.fill(WHITE)
            
            # Draw title
            title = self.font.render("Corridor Game", True, BLACK)
            title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 100))
            self.screen.blit(title, title_rect)
            
            # Draw buttons
            self.draw_button(self.buttons['pvp'], "Play vs Player", GRAY)
            self.draw_button(self.buttons['pve_simple'], "Play vs Simple AI", GRAY)
            self.draw_button(self.buttons['pve_super'], "Play vs Super AI", GRAY)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    for button_name, button_rect in self.buttons.items():
                        if button_rect.collidepoint(mouse_pos):
                            if button_name == 'pvp':
                                game = Game()
                                game.run_pvp()
                            elif button_name == 'pve_simple':
                                game = Game()
                                game.run_pve(SimpleAI())
                            elif button_name == 'pve_super':
                                game = Game()
                                game.run_pve(SuperAI())
            
            pygame.display.flip()

if __name__ == "__main__":
    menu = Menu()
    menu.run() 