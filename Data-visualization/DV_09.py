%matplotlib inline
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

url = "https://www.geeksforgeeks.org/how-to-automate-an-excel-sheet-in-python/?ref=feed"
html = urllib.request.urlopen(url)
htmlParse = BeautifulSoup(html, 'html.parser')
p = htmlParse.find("p").get_text()
print(p)

def Cvc(text):
    vowels = 'AEIOUaeiou'
    vowelcount = consonentcount = 0
    for char in text:
        if char.isalpha():
            if char in vowels:
                vowelcount += 1
            else:
                consonentcount += 1
    return vowelcount, consonentcount

vowelcount, consonentcount = Cvc(p)
print(f"Vowel count: {vowelcount}, Consonant count: {consonentcount}")

plt.pie([vowelcount, consonentcount], labels=['Vowels', 'Consonants'], autopct='%1.1f%%', startangle=90)
plt.title('Vowel and Consonant Distribution')
plt.show()
