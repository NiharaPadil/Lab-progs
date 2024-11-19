%matplotlib inline
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Correct URL format
url = "https://www.geeksforgeeks.org/how-to-automate-an-excel-sheet-in-python/?ref=feed"

# Fetch HTML from the URL
html = urllib.request.urlopen(url)

# Parse HTML using BeautifulSoup
htmlParse = BeautifulSoup(html, 'html.parser')

# Extract the first <p> tag and get its text
p = htmlParse.find("p").get_text()

# Print the extracted paragraph for verification
print(p)

# Function to count vowels and consonants
def Cvc(text):
    vowels = 'AEIOUaeiou'
    vowelcount = 0
    consonentcount = 0
   
    # Count vowels and consonants
    for char in text:
        if char.isalpha():  # Ensure the character is alphabetic
            if char in vowels:
                vowelcount += 1
            else:
                consonentcount += 1
    return vowelcount, consonentcount

# Get the counts of vowels and consonants
vowelcount, consonentcount = Cvc(p)

# Categories and values for pie chart
categories = ['Vowels', 'Consonants']
values = [vowelcount, consonentcount]

# Print the counts for verification
print(f"Vowel count: {vowelcount}, Consonant count: {consonentcount}")

# Create a pie chart
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title('Vowel and Consonant Distribution')
plt.show()
