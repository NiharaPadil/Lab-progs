#prog 2: NIMGAME
print("Nimgame!!\n we are having 12 tokens")
def getTokens(curTokens):
    global tokens
    print("How many tokens would you like to take?",end="")
    take =int(input())
    if (take<1 or take>3):
        print("Number must be between 1 and 3\n")
        getTokens(curTokens)
        return
    tokens=curTokens-take
    print('you take',take,'tokens')
    print(tokens,'tokens remaining\n')
def compTurn(curTokens):
    global tokens
    take =curTokens%4
    tokens=curTokens-take
    print('computer takes',take,'tokens')
    print(tokens,'tokens remaining.\n')
tokens=12
while(tokens>0):
    getTokens(tokens)
    compTurn(tokens)
print("Computer wins!")


#prog 3: WaterJug
from collections import defaultdict

jug1, jug2, aim = 5,3,4

#initialize dictionary with default value as false.
visited = defaultdict(lambda: False)

#Recursive function which prints the intermediate steps to reach the final solution and return Boolean value

def waterJugSolver(amt1, amt2):
    #Checks for our goal and returns true if achieved.
        if(amt1 == aim and amt2==0) or (amt2 == aim and amt1 == 0):
            print(amt1, amt2)
            return True
        if visited[(amt1, amt2)] == False:
            print(amt1, amt2)
            
            visited[(amt1, amt2)] = True
        
            return (waterJugSolver(0, amt2)or 
                    waterJugSolver(amt1, 0)or 
                    waterJugSolver(jug1, amt2)or
                    waterJugSolver(amt1, jug2)or 
                    waterJugSolver(amt1+min(amt2, (jug1-amt1)), amt2 - min(amt2, (jug1 -amt1)))or 
                    waterJugSolver(amt1 - min(amt1, (jug2 - amt1)), amt2+ min(amt1, (jug2 - amt2))))
        else:
            return False
        
print("Steps")
waterJugSolver(0,0)

#tictactoe vs comp
import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board, player):
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True

    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

def is_board_full(board):
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

def get_player_move():
    row = int(input("Enter row (0, 1, or 2): "))
    col = int(input("Enter column (0, 1, or 2): "))
    return row, col

def get_computer_move(board):
    empty_cells = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
    return random.choice(empty_cells)

def tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    while True:
        print_board(board)

        if current_player == 'X':
            row, col = get_player_move()
        else:
            print("Computer's move:")
            row, col = get_computer_move(board)

        if board[row][col] == ' ':
            board[row][col] = current_player

            if check_winner(board, current_player):
                print_board(board)
                if current_player == 'X':
                    print("Player X wins!")
                else:
                    print("Computer wins!")
                break

            if is_board_full(board):
                print_board(board)
                print("It's a tie!")
                break

            current_player = 'O' if current_player == 'X' else 'X'
        else:
            print("Cell already occupied. Try again.")

if __name__ == "__main__":
    tic_tac_toe()

#tictactoe 2 player
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board, player):
    # Check rows and columns
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True

    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

def is_board_full(board):
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

def tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    while True:
        print_board(board)

        row = int(input(f"Player {current_player}, enter row (0, 1, or 2): "))
        col = int(input(f"Player {current_player}, enter column (0, 1, or 2): "))

        if board[row][col] == ' ':
            board[row][col] = current_player

            if check_winner(board, current_player):
                print_board(board)
                print(f"Player {current_player} wins!")
                break

            if is_board_full(board):
                print_board(board)
                print("It's a tie!")
                break

            current_player = 'O' if current_player == 'X' else 'X'
        else:
            print("Cell already occupied. Try again.")

if __name__ == "__main__":
    tic_tac_toe()
