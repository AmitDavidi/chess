import pygame
import os
from pathlib import Path

cwd = os.getcwd()
images = os.path.join(cwd, "Resources")
HEIGHT = 700
WIDTH = 700
COLS = 8
SQUARE_SIZE = HEIGHT // 8
pygame.font.init()
WIN = pygame.display.set_mode((HEIGHT, WIDTH))
MOVES = pygame.Surface((HEIGHT, WIDTH))
constants = {
    'HEIGHT': HEIGHT,
    'WIDTH': WIDTH,
    'COLS': COLS,
    'SQUARE_SIZE': SQUARE_SIZE,
    'font': pygame.font.SysFont('Arial', 25),
    'Chess_Pieces': ['R', 'r', 'N', 'n', 'B', 'b', 'Q', 'q', 'K', 'k', 'P', 'p']
}

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BROWN = (113, 81, 57)
LIGHT_BROWN = (187, 159, 123)


# returns the opposite color of the piece
def op_color(piece):
    color = piece.Color
    if color == 'w':
        op_color = 'b'
    elif color == 'b':
        op_color = 'w'
    return op_color


# adds to legal moves that move
def check_move(board, color, x, y, i, j, piece):
    try:
        cur_square: Square = board.grid[x + i][y + j]
    except IndexError:
        return False
    if cur_square.same_color(color):
        return False
    board.legal_moves[(piece, cur_square)] = True
    cur_square.control[color] = True
    piece.Moves.append(cur_square)

    if cur_square.is_empty():
        # continue checking
        return True

    elif cur_square.can_attack(color):
        # stop here
        return False

    # good move continue looking
    return True


def check_left(Playing_board, color, x_pos, y_pos, piece):
    i = -1
    while x_pos + i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, i, 0, piece):
            i -= 1
        else:
            break


def check_right(Playing_board, color, x_pos, y_pos, piece):
    i = 1
    while x_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, i, 0, piece):
            i += 1
        else:
            break


def check_up(Playing_board, color, x_pos, y_pos, piece):
    i = -1
    while y_pos + i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, 0, i, piece):
            i -= 1
        else:
            break


def check_down(Playing_board, color, x_pos, y_pos, piece):
    i = 1
    while y_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, 0, i, piece):
            i += 1
        else:
            break


def check_right_diag(Playing_board, color, x_pos, y_pos, piece):
    i = 1
    while x_pos + i < COLS and y_pos - i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, i, -i, piece):
            i += 1
        else:
            break

    i = -1
    while x_pos + i >= 0 and y_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, i, -i, piece):
            i -= 1
        else:
            break


def check_left_diag(Playing_board, color, x_pos, y_pos, piece):
    i = 1
    while x_pos + i < COLS and y_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, i, i, piece):
            i += 1
        else:
            break

    i = -1
    while x_pos + i >= 0 and y_pos + i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, i, i, piece):
            i -= 1
        else:
            break


class Piece:
    def __init__(self, id, value, color, name, image, square=None, number=1):
        self.Id = id
        self.Value = value
        self.Color = color
        self.Number = number
        self.Name = name
        self.Image = image

        self.Square = square
        self.Moves = []

    def move_piece(self, target_square, board):
        if (self, target_square) in board.legal_moves.keys():
            target_square: Square
            # remove piece from square
            self.Square.Piece_on_Square = None
            # move piece to target square
            self.Square = target_square
            # update the piece on the new square
            target_square.Piece_on_Square = self
            self.Moves = []
            # success
            if hasattr(self, "moved_flag"):
                self.moved_flag = True
            return True
        else:
            return False

    def on_click(self, grid):
        self.Generate_Moves(grid)

        for square in self.Moves:
            square.draw_dot = True
            square.Color = (square.Color[0] * 1.2, square.Color[1] * 1.2, square.Color[2] * 1.2)

    def on_release(self):
        for square in self.Moves:
            square.draw_dot = False
            square.Color = square.Holder


class King(Piece):
    pass

    def __repr__(self):
        return str(self.Square) + self.Name + self.Color

    def Generate_Moves(self, Playing_board):
        grid = Playing_board.grid
        self.Moves = []
        Playing_board.legal_moves = {}
        move_range = [-1, 0, 1]
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        if color == 'w':
            op_color = 'b'
        elif color == 'b':
            op_color = 'w'

        for i in move_range:
            for j in move_range:

                if i == 0 and j == 0:
                    continue
                if 0 <= x_pos + i <= 7 and 0 <= y_pos + j <= 7:
                    cur_square: Square = grid[x_pos + i][y_pos + j]
                    cur_square.control[color] = True
                    # if the square is in control of the opposite color - continue
                    if op_color in cur_square.control:
                        continue
                    if cur_square.is_empty() or cur_square.can_attack(color):
                        self.Moves.append(cur_square)
                        Playing_board.legal_moves[(self, cur_square)] = True


class Quin(Piece):
    pass

    def Generate_Moves(self, Playing_board):
        self.Moves = []
        Playing_board.legal_moves = {}
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color

        check_left(Playing_board, color, x_pos, y_pos, self)
        check_right(Playing_board, color, x_pos, y_pos, self)
        check_up(Playing_board, color, x_pos, y_pos, self)
        check_down(Playing_board, color, x_pos, y_pos, self)
        check_right_diag(Playing_board, color, x_pos, y_pos, self)
        check_left_diag(Playing_board, color, x_pos, y_pos, self)


class Pawn(Piece):
    def __init__(self, id, value, color, name, image, square=None, number=1):
        super().__init__(id, value, color, name, image, square, number)
        self.moved_flag = False

    def Generate_Moves(self, Playing_board):
        grid = Playing_board.grid
        self.Moves = []
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        advance_squares = []
        attack_squares = []
        if color == 'w':
            for i in [-1, 1]:
                try:
                    attack_squares.append(grid[x_pos + i][y_pos - 1])
                except IndexError:
                    continue
            if grid[x_pos][y_pos - 1].is_empty():
                advance_squares.append(grid[x_pos][y_pos - 1])

                if self.moved_flag is False:
                    advance_squares.append(grid[x_pos][y_pos - 2])

        else:
            for i in [-1, 1]:
                try:
                    attack_squares.append(grid[x_pos + i][y_pos + 1])
                except IndexError:
                    continue

            if grid[x_pos][y_pos + 1].is_empty():
                advance_squares.append(grid[x_pos][y_pos + 1])

                if self.moved_flag is False:
                    advance_squares.append(grid[x_pos][y_pos + 2])

        for square in advance_squares:
            if square.is_empty():
                self.Moves.append(square)
                Playing_board.legal_moves[(self, square)] = True

        for square in attack_squares:
            square.control[color] = True
            if square.diff_color(color):
                self.Moves.append(square)
                Playing_board.legal_moves[(self, square)] = True


class Rook(Piece):

    def Generate_Moves(self, Playing_board):
        self.Moves = []
        Playing_board.legal_moves = {}
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        check_left(Playing_board, color, x_pos, y_pos, self)
        check_right(Playing_board, color, x_pos, y_pos, self)
        check_up(Playing_board, color, x_pos, y_pos, self)
        check_down(Playing_board, color, x_pos, y_pos, self)


class Knight(Piece):
    # if ((abs(i - self.Pos[0]) + abs(j - self.Pos[1])) == 3 and (i > 0) and (j > 0) and j < board_size + 1 and i < board_size + 1):
    def Generate_Moves(self, Playing_board):
        self.Moves = []
        Playing_board.legal_moves = {}
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= x_pos + i < COLS and 0 <= y_pos + j < COLS:
                    if abs(i) + abs(j) == 3:
                        check_move(Playing_board, color, x_pos, y_pos, i, j, self)


class Bishop(Piece):

    def Generate_Moves(self, Playing_board):
        self.Moves = []
        Playing_board.legal_moves = {}
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        check_right_diag(Playing_board, color, x_pos, y_pos, self)
        check_left_diag(Playing_board, color, x_pos, y_pos, self)


class Board:
    def __init__(self, window):
        self.window = window
        self.grid = []
        self.turn = None

        self.black_castle_quin = False
        self.black_castle_king = False
        self.white_castle_quin = False
        self.white_castle_king = False
        self.legal_moves = {}
        self.piece_nums = {}
        self.black_in_check = False
        self.white_in_check = False
        self.en_passant_square = None

        self.pieces = {'k': [King(1, float('inf'), 'b', 'k', pygame.image.load(os.path.join(images, "bk.png")))],
                       'K': [King(1, float('inf'), 'w', 'k', pygame.image.load(os.path.join(images, "wk.png")))],
                       'q': [Quin(2, 9, "b", 'q', pygame.image.load(os.path.join(images, 'bq.png')))],
                       'Q': [Quin(2, 9, "w", 'Q', pygame.image.load(os.path.join(images, 'wq.png')))],
                       'r': [Rook(3, 5, "b", 'r', pygame.image.load(os.path.join(images, 'br.png'))),
                             Rook(3, 5, "b", 'r', pygame.image.load(os.path.join(images, 'br.png')), number=2)],
                       'R': [Rook(3, 5, "w", 'R', pygame.image.load(os.path.join(images, 'wr.png'))),
                             Rook(3, 5, "w", 'R', pygame.image.load(os.path.join(images, 'wr.png')), number=2)],
                       'n': [Knight(4, 3, "b", 'n', pygame.image.load(os.path.join(images, 'bn.png'))),
                             Knight(4, 3, "b", 'n', pygame.image.load(os.path.join(images, 'bn.png')), number=2)],
                       'N': [Knight(4, 3, "w", 'N', pygame.image.load(os.path.join(images, 'wn.png'))),
                             Knight(4, 3, "w", 'N', pygame.image.load(os.path.join(images, 'wn.png')), number=2)],
                       'b': [Bishop(5, 3, "b", 'b', pygame.image.load(os.path.join(images, 'bb.png'))),
                             Bishop(5, 3, "b", 'b', pygame.image.load(os.path.join(images, 'bb.png')), number=2)],
                       'B': [Bishop(5, 3, "w", 'B', pygame.image.load(os.path.join(images, 'wb.png'))),
                             Bishop(5, 3, "w", 'B', pygame.image.load(os.path.join(images, 'wb.png')), number=2)],
                       'p': [Pawn(6, 1, "b", 'p', pygame.image.load(os.path.join(images, 'bp.png')), number=number) for
                             number in
                             range(1, 9)],
                       'P': [Pawn(6, 1, "w", 'P', pygame.image.load(os.path.join(images, 'wp.png')), number=number) for
                             number in
                             range(1, 9)]}

    def draw(self):
        for row in self.grid:
            for square in row:
                square: Square
                square.draw_square(self.window)
                if square.draw_dot:
                    pygame.draw.circle(WIN, BLACK, square.rect.center, 3)

    def King_in_Check(self):
        kings = self.pieces['k'] + self.pieces['K']
        for king in kings:
            if op_color(king) in king.Square.control:
                if king.Color == 'w':
                    self.white_in_check = True
                else:
                    self.black_in_check = True
            else:
                if king.Color == 'w':
                    self.white_in_check = False
                else:
                    self.black_in_check = False

        print('White: ', self.white_in_check, "Black: ", self.black_in_check)

    def legal_moves_reset(self):
        self.legal_moves = {}

    def reset_control(self):
        for row in self.grid:
            for square in row:
                square.control = {}

    def set_board(self):
        for i in range(COLS):
            self.grid.append([])

            for j in range(COLS):
                square = Square(X=i, Y=j)
                square: Square
                if i % 2 == j % 2:
                    square.Color = LIGHT_BROWN
                    square.Holder = LIGHT_BROWN
                else:
                    square.Color = DARK_BROWN
                    square.Holder = DARK_BROWN
                self.grid[i].append(square)

    def clear_board(self):
        self.piece_nums = {}
        for row in self.grid:
            for square in row:
                square: Square
                piece = square.Piece_on_Square
                if piece is not None:
                    piece: Piece
                    piece.Square = None
                    square.Piece_on_Square = None

    # not including times
    def parse_fen_code(self, code: str):
        self.clear_board()
        i = 0
        code_len = len(code)
        index = -1
        char = code[0]
        # first code pass
        # Piece Positions
        while char != ' ' and index + 1 < code_len:
            index += 1
            char = code[index]
            if char == '/':
                continue
            elif char.isnumeric():
                i += int(char)
                continue

            elif char.isalpha() and char in constants['Chess_Pieces']:
                j = i // COLS

                self.grid[i - (j * 8)][j].set_piece(char, self.pieces, self.piece_nums)
                i += 1

        if index == code_len - 1:
            return

        # advance to next pass - skip space
        index += 1
        # Who's Turn is it
        char = code[index]
        self.turn = char
        # advance to next code pass
        index += 1
        # first skip space - Castling Rights
        while char != ' ' and index + 1 < code_len:
            index += 1
            char = code[index]
            if char == 'k':
                self.black_castle_king = True
            elif char == 'K':
                self.white_castle_king = True
            elif char == 'Q':
                self.white_castle_quin = True
            elif char == 'q':
                self.black_castle_quin = True

        # advance to an passant skip space
        index += 1
        self.en_passant_square = code[index:]

    def grid_to_numpy_arr(self):
        pass

    def square_occupied(self, x, y):
        return self.grid[x][y].is_piece()

    def printf(self):
        pass

    def switch_turn(self):
        if self.turn == 'w':
            self.turn = 'b'
        elif self.turn == 'b':
            self.turn = 'w'

    def All_Generate_Moves(self):
        # before Generating moves again - clear the controls
        self.reset_control()
        self.legal_moves = {}

        for piece_stacks in self.pieces.values():
            for piece in piece_stacks:
                piece.Generate_Moves(self)


# X IS NUMBER OF COLUMN
class Square:
    def __init__(self, *, X, Y):
        self.Piece_on_Square: Piece = None
        self.Color = None
        self.X = X
        self.Y = Y
        self.rect = pygame.Rect(X * SQUARE_SIZE, Y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        self.control = {}
        self.Holder = None
        self.draw_dot = False

    def __repr__(self):
        return "(" + str(self.X) + "," + str(self.Y) + ")"

    def reset_control(self):
        self.control = {}

    def is_piece(self):
        return self.Piece_on_Square is not None

    def diff_color(self, color):
        if self.Piece_on_Square is None:
            return False
        if self.Piece_on_Square.Color != color:
            return True
        else:
            return False

    def same_color(self, color):
        if self.Piece_on_Square is not None:
            if self.Piece_on_Square.Color == color:
                return True
        else:
            return False

    def is_empty(self):
        return self.Piece_on_Square is None

    def can_attack(self, color):
        if self.Piece_on_Square is None:
            return False
        return self.Piece_on_Square.Color != color

    def draw_square(self, window):
        # draw background
        pygame.draw.rect(window, self.Color, self.rect)
        # text = constants['font'].render(str(self.X) + str(self.Y), True, BLACK)
        # window.blit(text, self.rect.center)
        # draw image \ text of piece
        if self.Piece_on_Square is not None:
            image_width = self.Piece_on_Square.Image.get_width()
            image_height = self.Piece_on_Square.Image.get_height()
            window.blit(self.Piece_on_Square.Image,
                        (self.rect.centerx - (image_width // 2), self.rect.centery - (image_height // 2)))

    # Gets a string that represents the Piece
    # the First setting of the piece
    def set_piece(self, piece_str, pieces, piece_num):
        try:
            piece_num[piece_str]
        except KeyError:
            piece_num[piece_str] = 1
        self.Piece_on_Square = pieces[piece_str][piece_num[piece_str] - 1]
        piece_num[piece_str] += 1
        self.Piece_on_Square.Square = self


def main(window):
    clock = pygame.time.Clock()

    board = Board(window)
    board.set_board()
    clicked = False
    running = True
    try:
        board.parse_fen_code('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KkQq')
    except IndexError:
        print("not enough pieces - Invalid FEN code")
    piece_to_move: Piece = None
    board.All_Generate_Moves()
    board.King_in_Check()
    while running:
        clock.tick(50)
        board.draw()
        pygame.display.update()
        x, y = pygame.mouse.get_pos()
        x = min(x // SQUARE_SIZE, COLS - 1)
        y = min(y // SQUARE_SIZE, COLS - 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    break

            if pygame.mouse.get_pressed(3)[0]:
                if not clicked:
                    clicked = True
                    clicked_square = board.grid[x][y]
                    if piece_to_move is None and board.square_occupied(x, y) and clicked_square.same_color(board.turn):
                        piece_to_move = clicked_square.Piece_on_Square
                        if piece_to_move.Color == board.turn:
                            # Generate that piece's legal moves and show them
                            piece_to_move.on_click(board)
                        else:
                            piece_to_move = None

                    elif piece_to_move is not None and x == piece_to_move.Square.X and y == piece_to_move.Square.Y:
                        piece_to_move.on_release()
                        piece_to_move = None

                    elif piece_to_move is not None and clicked_square.same_color(board.turn):
                        piece_to_move.on_release()
                        piece_to_move = clicked_square.Piece_on_Square
                        if piece_to_move.Color == board.turn:
                            # Generate that piece's legal moves and show them
                            piece_to_move.on_click(board)
                        else:
                            piece_to_move = None

                    elif piece_to_move is not None and clicked_square.Piece_on_Square != piece_to_move:
                        piece_to_move.on_release()
                        if piece_to_move.move_piece(clicked_square, board):
                            board.switch_turn()
                            piece_to_move = None
                            # update controlled squares
                            board.All_Generate_Moves()
                            board.King_in_Check()

            if event.type == pygame.MOUSEBUTTONUP:
                clicked = False


try:
    main(WIN)
except FileNotFoundError:
    print("your working directory is not set")
