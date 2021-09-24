import pygame
DARK_RED = (129, 31, 31)
RED = (130, 20, 20, 100)
DARK_GREEN = (10, 100, 10)
GREEN = (30, 215, 96, 250)
GREY = (215, 215, 215, 20)
DARK_GREY = (123, 85, 85)
BLACK = (25, 20, 20)
WHITE = (250, 250, 250)
BLUE = (51, 153, 255)
YELLOW = (250, 250, 153)
PURPLE = (128, 0, 128)
ORANGE = (255, 128, 0)
BRIGHT_GREEN = (102, 250, 250)

pygame.font.init()
# walak teeeeeel

class Button:
    def __init__(self, surface, on_click_func, color, x, y, width, height, text_color, text, text_size, exit_on_click, id=None):
        self.id = id
        self.color = color
        self.on_it_flag = False
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.text_size = text_size
        self.text_color = text_color
        self.surface = surface
        self.exit_on_click = exit_on_click

        # function to execute on - click -
        self.on_click_func = on_click_func

    # menu_button_1 = pygame.Rect(pause_width // 2 - 25, 30, menu_button_width, menu_button_height)
    # menu_button_2 = pygame.Rect(pause_width // 2 - 25, 130, menu_button_width, menu_button_height)

    def drawButton(self):  # , outline=None):
        # Call this method to draw the button on the screen
        # if outline:
        #    pygame.draw.rect(self.surface, outline, (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)
        pygame.draw.rect(self.surface, self.color, (self.x, self.y, self.width, self.height))
        if self.text != '':
            font = pygame.font.SysFont('arial', self.text_size)
            text = font.render(self.text, True, self.text_color)
            self.surface.blit(text, (
                self.x + (self.width // 2 - text.get_width() // 2),
                self.y + (self.height // 2 - text.get_height() // 2)))

    def update_text(self, new_text):
        self.text = new_text

    def isOver(self, pos, color=GREEN):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                self.color = color
                return True
        self.color = GREY
        return False

    def on_click(self):
        self.on_click_func()

# pass a Display surface, wanted height and wanted width- use draw_menu() to draw it.
class Menu:
    def __init__(self, surface, menu_height, menu_width, button_to_exit='m'):
        self.surface = surface  # Main Display surface --
        self.height = menu_height  # Height of Menu
        self.width = menu_width  # Width of Menu
        self.MAIN_DISPLAY_WIDTH = surface.get_width()  # The Width of the Main Display
        self.MAIN_DISPLAY_HEIGHT = surface.get_height()  # Ditto
        self.start_location_x = self.MAIN_DISPLAY_WIDTH - menu_width  # Start location - X of the Menu - Right Side
        self.end_location_y = self.MAIN_DISPLAY_HEIGHT - menu_height  # Height Location - Y of ^^^
        self.buttons = []
        self.pause_menu_surface = pygame.Surface((menu_width, menu_height))  # , pygame.SRCALPHA)
        self.Clock = pygame.time.Clock()
        self.button_to_exit = button_to_exit

    def draw_menu(self):
        self.pause_menu_surface.fill((50, 50, 50, 0))
        # animate entry
        # t = pause_width
        # while t >= 0:
        #    WIN.blit(pause_screen, (WIDTH - pause_width + t, 0))
        #    pygame.display.update()
        #    t -= 1

        running = True
        while running:
            self.Clock.tick(60)
            m_x, m_y = pygame.mouse.get_pos()
            m_x -= self.start_location_x
            pos = (m_x, m_y)

            for button in self.buttons:
                on_button = button.isOver(pos)
                if on_button:
                    button.on_it_flag = True
                    button.color = BLUE
                else:
                    button.on_it_flag = False
                    button.color = GREY

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if pygame.key.name(event.key) == self.button_to_exit:
                        running = False
                        break

                if pygame.mouse.get_pressed(3)[0]:
                    # if user clicked outside of the menu - exit the menu
                    if m_x < 0:
                        running = False
                        break

                    # if user clicked on a button - execute onclick and exit the menu
                    for button in self.buttons:
                        if button.on_it_flag:
                            button.on_click()
                            if button.exit_on_click:
                                pygame.event.clear()
                                return

            for button in self.buttons:
                button.drawButton()

            self.surface.blit(self.pause_menu_surface, (self.start_location_x, 0))
            pygame.display.update()

    def add_button(self, color, on_click_function=lambda: None, text='', text_color=BLACK, text_size=20, exit_on_click=True):
        x, y = self.width * 0.1, len(self.buttons) * (self.height * 0.115) + 10
        width, height = self.width * 0.8, self.height * 0.1

        new_button = Button(self.pause_menu_surface, on_click_function, color, x, y, width, height, text_color, text,
                            text_size, exit_on_click)
        self.buttons.append(new_button)
