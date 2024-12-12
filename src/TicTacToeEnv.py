import numpy as np
import os
import time

class TicTacToeEnv:
    def __init__(self, player):
        self.board = np.zeros(9)
        self.board_to_show = [''] * 9
        self.action_space = len(self.board)

    def reset(self):
        self.board = np.zeros(9)
        self.board_to_show = [''] * 9
        return self.board

    def step(self, action, player):
        moves = self.get_legal_moves()
        if action in moves:
            self.board[action] = player
            self.board_to_show[action] = 'X' if player == 1 else 'O'
            self.render()
            return self.check_game_over(player)
        raise ValueError("Forbidden move")  # Exceção corrigida

    def check_game_over(self, player):
        # Verifica linhas, colunas e diagonais
        for i in range(3):
            if (np.all(self.board[i*3:(i+1)*3] == player) or  # Linhas
                np.all(self.board[i::3] == player)):  # Colunas
                return self.board, 10, True

        # Verifica diagonais
        if (np.all(self.board[[0, 4, 8]] == player) or
            np.all(self.board[[2, 4, 6]] == player)):
            return self.board, 10, True

        # Verifica empate
        if not np.any(self.board == 0):
            return self.board, 5, True  # Empate

        return self.board, -1, False  # Jogo continua

    def get_legal_moves(self):
        return [i for i in range(len(self.board)) if self.board[i] == 0]

    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in range(3):
            print(" | ".join(
                [self.board_to_show[i * 3 + j] if self.board_to_show[i * 3 + j] != '' else f'{i * 3 + j}' for j in range(3)]
            ))
            if i < 2:
                print("---------")
        print('\n\n\n\n')
        time.sleep(0.5)

    def play(self, move, player):
        self.render()
        try:
            state, reward, done = self.step(move, player)
            self.render()
            if reward == 5:
                print("Empate!")
            elif reward == 10:
                print(f"Player {player} wins!" if reward > 0 else f"Player {-player} wins!")
            return state, reward, done
        except ValueError as e:
            print(f"Erro: {e}. Por favor, escolha um movimento válido.")
        except Exception as e:
            print(f"Erro inesperado: {e}")