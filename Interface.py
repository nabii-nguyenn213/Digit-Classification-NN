import pygame
from train import *

#constant
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Board:
    
    def __init__(self):
        self.board = self.make_board()
    
    def make_board(self):
        self.row = 28
        self.col = 28
        board = []
        for i in range(self.row):
            r = []
            for j in range(self.col):
                r.append(0)
            board.append(r)
        return board
    
    def fill_board(self, coors):
        for coor in coors:
            self.board[coor[0]][coor[1]] = 1
    
    def tonumpy(self):
        return np.array(self.board)
    
    def print_board(self):
        for i in self.board:
            print(i)

    def clear_board(self):
        self.board = self.make_board()


class PygameInterface:
    
    def __init__(self):
        self.board = Board()
        self.coor = []
        self.y_pred = 0
    
    def init(self):
        pygame.init()
        win_size = (1000, 700)
        self.window = pygame.display.set_mode(win_size)
        self.window.fill('cadetblue4')
        pygame.display.set_caption("Handwritten Digit Recognition")
        
    def run(self):
        running = True
        while running:
            self.mouse_x, self.mouse_y = pygame.mouse.get_pos()
            self.draw_grid()
            self.button()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if pygame.mouse.get_pressed()[0]:
                    if (14 < self.mouse_x < 686) and (14 < self.mouse_y < 686):
                        self.draw()
                        self.board.fill_board(self.coor)
                        # self.board.print_board()
                        # print()
        
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (870 <= self.mouse_x <= 970) and (640 <= self.mouse_y <= 670):
                        # print('clear')
                        self.clear_board()
                    if (720 <= self.mouse_x <= 820) and (640 <= self.mouse_y <= 670):
                        # print('predict')
                        img = self.board.tonumpy()
                        print(img)
                        img = img.reshape(784, 1)
                        self.y_pred = pred(img)
                        print(self.y_pred)
                        
                        
            pygame.display.flip()
    
    def button(self):
        font = pygame.font.SysFont('sans', 25)
        clear_text = font.render('clear', True, BLACK)
        predict_text = font.render('predict', True, BLACK)
        pygame.draw.rect(self.window, WHITE, (870, 640, 100, 30)) # clear button
        self.window.blit(clear_text, (895, 640))
        pygame.draw.rect(self.window, WHITE, (720, 640, 100, 30))
        self.window.blit(predict_text, (735, 640))
        
        predict_number = font.render(str(self.y_pred), True, BLACK)
        predict_text_ = font.render('predict :', True, BLACK)
        pygame.draw.rect(self.window, WHITE, (750, 50, 200, 100))
        self.window.blit(predict_text_, (765, 80))
        self.window.blit(predict_number, (850 , 82))
        
        
    
    def draw(self):
        col = (self.mouse_x - 14) // 24
        row = (self.mouse_y - 14) // 24
        # print(col, row)
        col1 = max(col - 1, 0)
        row1 = max(row - 1, 0)
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col), (14 + 24 * row), 24, 24))
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col1), (14 + 24 * row1), 24, 24))
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col1), (14 + 24 * row), 24, 24))
        pygame.draw.rect(self.window, WHITE, ((14 + 24 * col), (14 + 24 * row1), 24, 24))
        
        if (row, col) not in self.coor:
            self.coor.append((row, col))
        if (row1, col1) not in self.coor:
            self.coor.append((row1, col1))
        if (row1, col) not in self.coor:
            self.coor.append((row1, col))
        if (row, col1) not in self.coor:
            self.coor.append((row, col1))
            
    
    def clear_board(self):
        self.window.fill('cadetblue4')
        self.draw_grid()
        self.coor = []
        self.board.clear_board()
        
    
    def draw_grid(self):
        for r in range(self.board.row+1):
            pygame.draw.line(self.window, WHITE, (r * 24 + 14, 14), (r * 24 + 14, 686))
            pygame.draw.line(self.window, WHITE, (14, r * 24 + 14), (686, r * 24 + 14))

'''
training
'''
print("waiting training process...")
fit(lr = 0.001, batch_size=32, epochs=10000)
print("training done!")

window = PygameInterface()
window.init()
window.run()