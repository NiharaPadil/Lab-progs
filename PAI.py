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
