import random
class Crossword:
    def __init__(self, words):
        self.words = [word.upper() for word in words]
        self.grid = [[' ' for _ in range(max(len(word) for word in self.words))] for _ in range(max(len(word) for word in self.words))]
    def generate_grid(self):
        for word in self.words:
            direction = random.choice(['horizontal', 'vertical'])
            row = random.randint(0, len(self.grid) - len(word)) if direction == 'vertical' else random.randint(0, len(self.grid) - 1)
            col = random.randint(0, len(self.grid[0]) - len(word)) if direction == 'horizontal' else random.randint(0, len(self.grid[0]) - 1)
            for i, letter in enumerate(word):
                if direction == 'vertical':
                    self.grid[row + i][col] = letter
                else:
                    self.grid[row][col + i] = letter if word.isupper() else letter.lower()
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == ' ':
                    self.grid[i][j] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    def find_words(self):
        found_words = []
        for word in self.words:
            if any([self.check_horizontal(word, i, j) or self.check_vertical(word, i, j) or self.check_diagonal(word, i, j) for i in range(len(self.grid)) for j in range(len(self.grid[i]))]):
                found_words.append(word)
        return found_words

    def check_horizontal(self, word, row, col):
        return all([self.grid[row][col + i].upper() == word[i] for i in range(len(word))]) if col + len(word) <= len(self.grid[0]) else False
    def check_vertical(self, word, row, col):
        return all([self.grid[row + i][col].upper() == word[i] for i in range(len(word))]) if row + len(word) <= len(self.grid) else False
    def check_diagonal(self, word, row, col):
        if row + len(word) <= len(self.grid) and col + len(word) <= len(self.grid[0]):
            if all([self.grid[row + i][col + i].upper() == word[i] for i in range(len(word))]):
                return True
        if row - len(word) >= -1 and col + len(word) <= len(self.grid[0]):
            if all([self.grid[row - i][col + i].upper() == word[i] for i in range(len(word))]):
                return True
        return False

    def print_grid(self):
        [print(' '.join(row)) for row in self.grid]
if __name__ == "__main__":
    words_to_find = ['HELLO', 'GOOGLE', 'CROSSWORD']
    crossword = Crossword(words_to_find)
    crossword.generate_grid()
    crossword.print_grid()
    found_words = crossword.find_words()
    for word in words_to_find:
        if word in found_words:
            print(f"Found {word}")
        else:
            print(f"{word} not found")
