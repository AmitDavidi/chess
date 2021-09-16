import os

import pygame

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
def opposite_piece_color(piece):
    color = piece.Color
    if color == 'w':
        opposite_color = 'b'
    else:
        opposite_color = 'w'

    return opposite_color




# adds to legal moves that move
def check_move(board, color, x, y, i, j, piece, just_update_moves):
    try:
        cur_square: Square = board.grid[x + i][y + j]
    except IndexError:
        return False
    cur_square.control[color] = True

    if cur_square.same_color(color):
        return False

    # redundant ??
    if just_update_moves:
        if cur_square.can_attack(color):
            return False

    if not just_update_moves:
        board.legal_moves[(piece, cur_square)] = True

        piece.Moves.append(cur_square)

        if cur_square.is_empty():
            # continue checking
            return True

        elif cur_square.can_attack(color):
            # stop here
            return False

        # good move continue looking
        return True
    return True


def check_left(Playing_board, color, x_pos, y_pos, piece, just_update_moves):
    i = -1
    while x_pos + i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, i, 0, piece, just_update_moves):
            i -= 1
        else:
            break


def check_right(Playing_board, color, x_pos, y_pos, piece, just_update_moves):
    i = 1
    while x_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, i, 0, piece, just_update_moves):
            i += 1
        else:
            break


def check_up(Playing_board, color, x_pos, y_pos, piece, just_update_moves):
    i = -1
    while y_pos + i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, 0, i, piece, just_update_moves):
            i -= 1
        else:
            break


def check_down(Playing_board, color, x_pos, y_pos, piece, just_update_moves):
    i = 1
    while y_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, 0, i, piece, just_update_moves):
            i += 1
        else:
            break


def check_right_diag(Playing_board, color, x_pos, y_pos, piece, just_update_moves):
    i = 1
    while x_pos + i < COLS and y_pos - i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, i, -i, piece, just_update_moves):
            i += 1
        else:
            break

    i = -1
    while x_pos + i >= 0 and y_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, i, -i, piece, just_update_moves):
            i -= 1
        else:
            break


def check_left_diag(Playing_board, color, x_pos, y_pos, piece, just_update_moves):
    i = 1
    while x_pos + i < COLS and y_pos + i < COLS:
        if check_move(Playing_board, color, x_pos, y_pos, i, i, piece, just_update_moves):
            i += 1
        else:
            break

    i = -1
    while x_pos + i >= 0 and y_pos + i >= 0:
        if check_move(Playing_board, color, x_pos, y_pos, i, i, piece, just_update_moves):
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

    def kill(self):
        if self.Square is not None:
            self.Square.Piece_on_Square = None
        self.Square = None

    def move_sim(self, target):
        piece_on_target = target.Piece_on_Square
        if piece_on_target is not None:
            piece_on_target.Square = None

        target.Piece_on_Square = None
        if self.Square is not None:
            self.Square.Piece_on_Square = None
        self.Square = target
        target.Piece_on_Square = self

    def move_piece(self, target_square, board, castling=False):
        if (self, target_square) in board.legal_moves.keys() or castling:
            target_square: Square
            # remove enemy piece from target square
            piece_on_target = target_square.Piece_on_Square
            if piece_on_target is not None:
                piece_on_target.kill()

            # remove piece from square
            if self.Square is not None:
                self.Square.Piece_on_Square = None
            # move piece to target square
            self.Square = target_square
            # update the piece on the new square
            target_square.Piece_on_Square = self
            self.Moves = []
            # success
            if hasattr(self, "moved_flag"):
                if self.Square.Y in [0, 7]:
                    square = self.Square
                    self.kill()
                    if self.Color == 'b':
                        piece_to_set_after_pawn_advance = 'q'
                    else:
                        piece_to_set_after_pawn_advance = 'Q'
                    square.set_piece(piece_to_set_after_pawn_advance, board.pieces, board.piece_nums)


                self.moved_flag = True
            if hasattr(self, "rook_type"):
                if self.rook_type == 'king':
                    if self.Color == 'b':
                        board.black_castle_king = False
                    elif self.Color == 'w':
                        board.white_castle_king = False
                elif self.rook_type == 'queen':
                    if self.Color == 'b':
                        board.black_castle_quin = False
                    elif self.Color == 'w':
                        board.white_castle_quin = False

            if hasattr(self, "king_moved"):
                if self.king_moved == False:
                    if self.Square.X == 1 or self.Square.X == 5 or self.Square.X == 2 or self.Square.X == 6:
                        board.move_rook(self)


                if self.Color == 'b':
                    board.black_castle_king = False
                    board.black_castle_quin = False
                if self.Color == 'w':
                    board.white_castle_quin = False
                    board.white_castle_king = False
                self.king_moved = True


            return True
        else:
            return False

    def update_moves(self, board):

        if self.Color == 'b':
            king = board.pieces['k'][0]
        elif self.Color == 'w':
            king = board.pieces['K'][0]

        to_pop = []
        for move in self.Moves:
            target_square = move
            # save the original piece on target
            piece_on_target: Piece = move.Piece_on_Square

            # move self - to target
            self.move_sim(move)
            # update controls - check if king in check
            board.Update_Square_Controllers()

            if king.in_check():
                to_pop.append(move)

            # return original piece to target
            if piece_on_target is not None:
                piece_on_target.move_sim(target_square)

        for move_to_remove in to_pop:
            self.Moves.remove(move_to_remove)
            board.legal_moves.pop((self, move_to_remove))

    def on_click(self, board):
        board.reset_control()

        self.Generate_Moves(board)
        # save the square of self
        first_square = self.Square
        # move simulations of self
        self.update_moves(board)
        # return self to it's original square
        self.move_sim(first_square)

        for square in self.Moves:
            square.draw_dot = True
            square.Color = (square.Color[0] * 1.2, square.Color[1] * 1.2, square.Color[2] * 1.2)

    def on_release(self):
        for square in self.Moves:
            square.draw_dot = False
            square.Color = square.Holder


class King(Piece):
    def __init__(self, id, value, color, name, image, square=None, number=1):
        super().__init__(id, value, color, name, image, square, number)
        self.king_moved = False



    def __repr__(self):
        return str(self.Square) + self.Name + self.Color
    def can_castle(self, board, side: str = None):
        # black
        if self.Color == 'b':
            # quin side
            if side == 'quin':
                return board.black_castle_quin
            else: # king side
                return board.black_castle_king
        # white
        else:
            if side == 'king': # king side
                return board.white_castle_king
            else: # quin side
                return board.white_castle_quin

    def Generate_Moves(self, Playing_board, just_update_squares=False):
        grid = Playing_board.grid
        if not just_update_squares:
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

        flipped = Playing_board.flip
        # True - K || Q
        # False - Q || K

        if not just_update_squares:
            # if not in check - check castle rights
            if not self.king_moved:
                if not self.in_check():
                    # check king side castle

                    if self.can_castle(Playing_board, 'king'):
                        # check if the way is free
                        if flipped:
                            if color == 'b':
                                if grid[2][7].is_empty() and grid[1][7].is_empty() and not (op_color in grid[1][7].control) and not (op_color in grid[2][7].control):
                                    self.Moves.append(grid[1][7])
                                    Playing_board.legal_moves[(self, grid[1][7])] = True

                            else:
                                if grid[2][0].is_empty() and grid[1][0].is_empty() and not (
                                        op_color in grid[1][0].control) and not (op_color in grid[2][0].control):
                                    self.Moves.append(grid[1][0])
                                    Playing_board.legal_moves[(self, grid[1][0])] = True

                        else:
                            if color == 'b':

                                if grid[5][0].is_empty() and grid[6][0].is_empty() and not (
                                        op_color in grid[5][0].control) and not (op_color in grid[6][0].control):
                                    self.Moves.append(grid[6][0])
                                    Playing_board.legal_moves[(self, grid[6][0])] = True

                            else:

                                if grid[5][7].is_empty() and grid[6][7].is_empty() and not (
                                        op_color in grid[5][7].control) and not (op_color in grid[6][7].control):

                                    self.Moves.append(grid[6][7])
                                    Playing_board.legal_moves[(self, grid[6][7])] = True

                    # check queen side castle
                    if self.can_castle(Playing_board, 'queen'):
                        # check if the way is free
                        if flipped:
                            if color == 'b':
                                if grid[4][7].is_empty() and grid[5][7].is_empty() and grid[6][7].is_empty() and not (
                                        op_color in grid[4][7].control) and not (op_color in grid[5][7].control) and not (op_color in grid[6][7].control):
                                    self.Moves.append(grid[5][7])
                                    Playing_board.legal_moves[(self, grid[5][7])] = True

                            else:
                                if grid[4][0].is_empty() and grid[5][0].is_empty() and grid[6][0].is_empty() and not (
                                        op_color in grid[4][0].control) and not (op_color in grid[5][0].control) and not (op_color in grid[6][0].control):
                                    self.Moves.append(grid[5][0])
                                    Playing_board.legal_moves[(self, grid[5][0])] = True

                        else:
                            if color == 'b':
                                if grid[1][0].is_empty() and grid[2][0].is_empty() and grid[3][0].is_empty() and not (
                                        op_color in grid[1][0].control) and not (op_color in grid[2][0].control) and not (op_color in grid[3][0].control):
                                    self.Moves.append(grid[2][0])
                                    Playing_board.legal_moves[(self, grid[2][0])] = True

                            else:
                                if grid[1][7].is_empty() and grid[2][7].is_empty() and grid[3][7].is_empty() and not (
                                        op_color in grid[1][7].control) and not (op_color in grid[2][7].control) and not (op_color in grid[3][7].control):
                                    self.Moves.append(grid[2][7])
                                    Playing_board.legal_moves[(self, grid[2][7])] = True

        for i in move_range:
            for j in move_range:

                if i == 0 and j == 0:
                    continue
                if 0 <= x_pos + i <= 7 and 0 <= y_pos + j <= 7:
                    cur_square: Square = grid[x_pos + i][y_pos + j]
                    cur_square.control[color] = True
                    if not just_update_squares:
                        # if the square is in control of the opposite color - continue
                        if op_color in cur_square.control:
                            continue
                        if cur_square.is_empty() or cur_square.can_attack(color):
                            self.Moves.append(cur_square)
                            Playing_board.legal_moves[(self, cur_square)] = True


    def in_check(self):
        if opposite_piece_color(self) in self.Square.control:
            return True
        return False


class Quin(Piece):
    pass

    def Generate_Moves(self, Playing_board, just_update_squares=False):
        if not just_update_squares:
            self.Moves = []
            Playing_board.legal_moves = {}
        if self.Square is None:
            return
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color

        check_left(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_right(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_up(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_down(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_right_diag(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_left_diag(Playing_board, color, x_pos, y_pos, self, just_update_squares)


class Pawn(Piece):
    def __init__(self, id, value, color, name, image, square=None, number=1):
        super().__init__(id, value, color, name, image, square, number)
        self.moved_flag = False

    def Generate_Moves(self, Playing_board, just_update_squares=False):
        grid = Playing_board.grid
        if not just_update_squares:
            self.Moves = []
            Playing_board.legal_moves = {}

        x_pos = self.Square.X
        y_pos = self.Square.Y

        color = self.Color

        advance_squares = []
        attack_squares = []
        flip = Playing_board.flip

        if flip:
            flip_color = opposite_piece_color(self)
        else:
            flip_color = color

        if flip_color == 'w':
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

            try:
                if grid[x_pos][y_pos + 1].is_empty():
                    advance_squares.append(grid[x_pos][y_pos + 1])

                    if self.moved_flag is False:
                        advance_squares.append(grid[x_pos][y_pos + 2])
            except IndexError:
                pass

        if not just_update_squares:
            for square in advance_squares:
                if square.is_empty():
                    self.Moves.append(square)
                    Playing_board.legal_moves[(self, square)] = True

            for square in attack_squares:
                square.control[color] = True
                if square.diff_color(color):
                    self.Moves.append(square)
                    Playing_board.legal_moves[(self, square)] = True
        else:
            for square in attack_squares:
                square.control[color] = True


class Rook(Piece):
    def __init__(self, id, value, color, name, image, square=None, number=1):
        super().__init__(id, value, color, name, image, square, number)
        self.rook_moved = False
        self.rook_type = None

    def Generate_Moves(self, Playing_board, just_update_squares=False):
        if not just_update_squares:
            self.Moves = []
            Playing_board.legal_moves = {}

        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        check_left(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_right(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_up(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_down(Playing_board, color, x_pos, y_pos, self, just_update_squares)


class Knight(Piece):
    # if ((abs(i - self.Pos[0]) + abs(j - self.Pos[1])) == 3 and (i > 0) and (j > 0) and j < board_size + 1 and i <
    # board_size + 1):
    def Generate_Moves(self, Playing_board, just_update_squares=False):
        if not just_update_squares:
            self.Moves = []
            Playing_board.legal_moves = {}
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= x_pos + i < COLS and 0 <= y_pos + j < COLS:
                    if abs(i) + abs(j) == 3:
                        check_move(Playing_board, color, x_pos, y_pos, i, j, self, just_update_squares)


class Bishop(Piece):

    def Generate_Moves(self, Playing_board, just_update_squares=False):
        if not just_update_squares:
            self.Moves = []
            Playing_board.legal_moves = {}
        x_pos = self.Square.X
        y_pos = self.Square.Y
        color = self.Color
        check_right_diag(Playing_board, color, x_pos, y_pos, self, just_update_squares)
        check_left_diag(Playing_board, color, x_pos, y_pos, self, just_update_squares)


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
        self.flip = False
        self.made_ill = False
        self.piece_nums = {}
        self.black_in_check = False
        self.white_in_check = False
        self.en_passant_square = None

        self.pieces = {'k': [King(1, float('inf'), 'b', 'k', pygame.image.load(os.path.join(images, "bk.png")))],
                       'K': [King(1, float('inf'), 'w', 'k', pygame.image.load(os.path.join(images, "wk.png")))],
                       'q': [Quin(2, 9, "b", 'q', pygame.image.load(os.path.join(images, 'bq.png')), number=number) for
                             number in range(1, 9)],
                       'Q': [Quin(2, 9, "w", 'Q', pygame.image.load(os.path.join(images, 'wq.png')), number=number) for
                             number in range(1, 9)],
                       'r': [Rook(3, 5, "b", 'r', pygame.image.load(os.path.join(images, 'br.png')), number=number) for
                             number in range(1, 9)],
                       'R': [Rook(3, 5, "w", 'R', pygame.image.load(os.path.join(images, 'wr.png')), number=number) for
                             number in range(1, 9)],
                       'n': [Knight(4, 3, "b", 'n', pygame.image.load(os.path.join(images, 'bn.png')), number=number)
                             for number in range(1, 9)],
                       'N': [Knight(4, 3, "w", 'N', pygame.image.load(os.path.join(images, 'wn.png')), number=number)
                             for number in range(1, 9)],
                       'b': [Bishop(5, 3, "b", 'b', pygame.image.load(os.path.join(images, 'bb.png')), number=number)
                             for number in range(1, 9)],
                       'B': [Bishop(5, 3, "w", 'B', pygame.image.load(os.path.join(images, 'wb.png')), number=number)
                             for number in range(1, 9)],
                       'p': [Pawn(6, 1, "b", 'p', pygame.image.load(os.path.join(images, 'bp.png')), number=number) for
                             number in range(1, 9)],
                       'P': [Pawn(6, 1, "w", 'P', pygame.image.load(os.path.join(images, 'wp.png')), number=number) for
                             number in
                             range(1, 9)]}

    def draw(self):

        col_range = range(COLS)
        for i in col_range:
            # for row in self.grid:
            for j in col_range:
                # for square in row:
                square: Square = self.grid[i][j]
                square.draw_square(self.window)
                if square.draw_dot:
                    pygame.draw.circle(WIN, BLACK, square.rect.center, 3)

    def King_in_Check(self):
        kings = self.pieces['k'] + self.pieces['K']
        for king in kings:
            if opposite_piece_color(king) in king.Square.control:
                if king.Color == 'w':
                    self.white_in_check = True
                else:
                    self.black_in_check = True
            else:
                if king.Color == 'w':
                    self.white_in_check = False
                else:
                    self.black_in_check = False

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
                piece = square.Piece_on_Square
                if piece is not None:
                    piece: Piece
                    piece.Square = None
                    square.Piece_on_Square = None

    # not including times
    def parse_fen_code(self, code: str):
        last = None
        for char in code:
            if last == ' ':
                if char == 'w':
                    flip = False
                elif char == 'b':
                    flip = True
            last = char

        self.flip = flip
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
                if not flip:
                    self.grid[i - (j * 8)][j].set_piece(char, self.pieces, self.piece_nums)
                else:
                    self.grid[abs(i - (j * 8) - 7)][abs(j - 7)].set_piece(char, self.pieces, self.piece_nums)
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
        self.fix_pawns()

    # fixes the pawns' first move rights - double advance - after parsing a FEN code
    def fix_pawns(self):
        if self.flip:
            black_line = 6
            white_line = 1
        else:
            black_line = 1
            white_line = 6

        for black_pawn in self.pieces['p']:
            if black_pawn.Square is not None:
                if black_pawn.Square.Y != black_line:
                    black_pawn.moved_flag = True

        for white_pawn in self.pieces['P']:
            if white_pawn.Square is not None:
                if white_pawn.Square.Y != white_line:
                    white_pawn.moved_flag = True

    def grid_to_numpy_arr(self):
        pass


    def update_rook_types(self):

        for rook in self.pieces['r'][0:2] + self.pieces['R'][0:2]:
            rook: Rook
            # if not flipped - left side is the queen side - x = 0
            if not self.flip:
                if rook.Square.X == 0:
                    rook.rook_type = 'queen'

                elif rook.Square.X == COLS-1:
                    rook.rook_type = 'king'
            else:
                if rook.Square.X == 0:
                    rook.rook_type = 'king'
                elif rook.Square.X == COLS-1:
                    rook.rook_type = 'queen'

    # given a castling command - move the correct rook.
    def move_rook(self, king):
        # X pos depends on flip
        if self.flip:
            if king.Square.X == 1:
                if self.turn == 'b':
                        # locate rook
                        for black_rook in self.pieces['r']:
                            black_rook: Rook
                            if black_rook.rook_type == 'king':
                                black_rook.move_piece(self.grid[2][7], self, castling=True)

                if self.turn == 'w':
                    # locate rook
                    for white_rook in self.pieces['R']:
                        white_rook: Rook
                        if white_rook.rook_type == 'king':
                            white_rook.move_piece(self.grid[2][0], self, castling=True)



            elif king.Square.X == 5:
                if self.turn == 'b':

                    # locate rook
                    for black_rook in self.pieces['r']:
                        black_rook: Rook
                        if black_rook.rook_type == 'queen':

                            black_rook.move_piece(self.grid[4][7], self, castling=True)

                elif self.turn == 'w':
                    # locate rook
                    for white_rook in self.pieces['R']:
                        white_rook: Rook
                        if white_rook.rook_type == 'queen':
                            white_rook.move_piece(self.grid[4][0], self, castling=True)

        else:

            if king.Square.X == 2:
                if self.turn == 'b':
                        # locate rook
                        for black_rook in self.pieces['r']:
                            black_rook: Rook
                            if black_rook.rook_type == 'queen':

                                black_rook.move_piece(self.grid[3][0], self, castling=True)

                if self.turn == 'w':

                    # locate rook
                    for white_rook in self.pieces['R']:
                        white_rook: Rook
                        if white_rook.rook_type == 'queen':

                            white_rook.move_piece(self.grid[3][7], self, castling=True)



            elif king.Square.X == 6:
                if self.turn == 'b':

                    # locate rook
                    for black_rook in self.pieces['r']:
                        black_rook: Rook
                        if black_rook.rook_type == 'king':

                            black_rook.move_piece(self.grid[5][0], self, castling=True)

                elif self.turn == 'w':
                    # locate rook
                    for white_rook in self.pieces['R']:
                        white_rook: Rook
                        if white_rook.rook_type == 'king':
                            white_rook.move_piece(self.grid[5][7], self, castling=True)



    def square_occupied(self, x, y):
        return self.grid[x][y].is_piece()

    def switch_turn(self):
        if self.turn == 'w':
            self.turn = 'b'
        elif self.turn == 'b':
            self.turn = 'w'

    def Update_Square_Controllers(self):
        # before Generating moves again - clear the controls
        self.reset_control()

        for piece_stacks in self.pieces.values():
            for piece in piece_stacks:
                if piece.Square is not None:
                    piece.Generate_Moves(self, just_update_squares=True)


# X IS NUMBER OF COLUMN
class Square:
    def __init__(self, *, X, Y):
        self.Piece_on_Square = None
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

def print_castle(board):
    #print("bk: ", board.black_castle_king, "bq: ", board. black_castle_quin, "wk: ", board. white_castle_king, "wq: ", board.white_castle_quin)
    pass

def main(window):
    clock = pygame.time.Clock()

    board = Board(window)
    board.set_board()
    clicked = False
    running = True
    try:
        # board.parse_fen_code('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KkQq')
        board.parse_fen_code('r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KkQq')
        # board.parse_fen_code('8/8/1K5q/8/4B1Q1/8/8/k7 w KkQq')
        # board.parse_fen_code('8/8/2k5/8/8/4b3/3K4/8 w KkQq')
        # board.parse_fen_code('8/6K1/1p1B1RB1/8/2Q5/2n1kP1N/3b4/4n3 w')
        # board.parse_fen_code('2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPKPP1/RNB2r2 b KQkq')
        #board.parse_fen_code("2kr1b1r/ppqnp2p/2p1Qp1P/3P4/3P4/2N5/PPP2PP1/R1B1KBNR b KQ")
        # board.parse_fen_code("8/4PPPPP/8/8/1k2K3/8/ppppp3/8 w")

    except IndexError:
        print("Invalid FEN code")

    board.update_rook_types()
    piece_to_move: Piece = None
    board.Update_Square_Controllers()
    board.King_in_Check()

    while running:
        print_castle(board)
        clock.tick(60)
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

                    if piece_to_move is None and clicked_square.same_color(board.turn):
                        # update piece to move
                        piece_to_move = clicked_square.Piece_on_Square
                        # Generate that piece's legal moves and show them
                        piece_to_move.on_click(board)

                    elif piece_to_move is not None:
                        # if clicked on same square - release the piece
                        if clicked_square == piece_to_move.Square:
                            piece_to_move.on_release()
                            piece_to_move = None

                        # if you clicked on a square that has a piece on it, and this piece is of the
                        # SAME color - pick it, instead of the current piece
                        elif clicked_square.same_color(board.turn):
                            piece_to_move.on_release()
                            piece_to_move = clicked_square.Piece_on_Square
                            # Generate that piece's legal moves and show them
                            piece_to_move.on_click(board)

                        # you clicked on a square that is not the same color, and
                        # not the same square - move to that square!
                        else:
                            # clear the drawings of the piece's moves
                            piece_to_move.on_release()
                            # try to move the piece
                            if piece_to_move.move_piece(clicked_square, board):
                                board.switch_turn()
                                piece_to_move = None
                                # update controlled squares
                                board.King_in_Check()
                            else:
                                # whoops - clicked on illegal square - don't move - release the piece
                                piece_to_move.on_release()
                                piece_to_move = None

            if event.type == pygame.MOUSEBUTTONUP:
                clicked = False


try:
    main(WIN)
except FileNotFoundError:
    print("your working directory is not set")
