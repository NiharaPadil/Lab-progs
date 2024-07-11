import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv('enjoysport.csv'))

concepts = np.array(data.iloc[:, 0:-1])
print("Concepts:\n", concepts)

target = np.array(data.iloc[:, -1])
print("Target:\n", target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print("Specific_h:", specific_h)
    
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("General_h:", general_h)
    
    # Training the model
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
            print(f"Specific_h after instance {i+1}:", specific_h)
        
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
            print(f"General_h after instance {i+1}:", general_h)
    
    indices = [i for i, val in enumerate(general_h) if val == ['?' for _ in range(len(specific_h))]]
    for i in indices:
        general_h.remove(['?' for _ in range(len(specific_h))])
    
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_h:\n", s_final)
print("Final General_h:\n", g_final)
