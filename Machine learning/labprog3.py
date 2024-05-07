def infogain(P,N):
    import math
    return -P/(P+N) * math.log2(P/(P+N))- N/(P+N) *math.log2(N/(P+N))
def insertnode(tree, addto,node):
    for k,v in tree.items():
        if isinstance(v,dict):
            tree[k]=insertnode(v,addto,node)
    if addto in tree:
        if isinstance(tree[addto],dict):
            tree[addto][node]='None'
        else:
            tree[addto]={node: 'None'}
    return tree
def insertconcept(tree,addto,node):
    for k,v in tree.items():
        if isinstance(v,dict):
            tree[k]=insertconcept(v,addto,node)
    if addto in tree:
        tree[addto]=node
    return tree
def getnextnode(data, attributelist, concept, conceptvals, tree, addto):
    total = data.shape[0]
    if total == 0:
        return tree
    
    countc = {}
    for cval in conceptvals:
        datacc = data[data[concept] == cval]
        countc[cval] = datacc.shape[0]
    
    if countc[conceptvals[0]] == 0:
        tree = insertconcept(tree, addto, conceptvals[1])
        return tree
    if countc[conceptvals[1]] == 0:
        tree = insertconcept(tree, addto, conceptvals[0])
        return tree
    
    classentropy = infogain(countc[conceptvals[0]], countc[conceptvals[1]])
    
    attr = {}
    for a in attributelist:
        attr[a] = list(set(data[a]))
    
    attrcount = {}
    entropyattr = {}
    for att in attr: 
        for vals in attr[att]:
            for c in conceptvals:
                idata = data[data[att] == vals]
                dataatt = idata[idata[concept] == c]
                attrcount[c] = dataatt.shape[0]
            
            totalinfo = attrcount[conceptvals[0]] + attrcount[conceptvals[1]]
            if attrcount[conceptvals[0]] == 0 or attrcount[conceptvals[1]] == 0:
                info = 0
            else:
                info = infogain(attrcount[conceptvals[0]], attrcount[conceptvals[1]])
            
            if att not in entropyattr:
                entropyattr[att] = (totalinfo / total) * info
            else:
                entropyattr[att] += (totalinfo / total) * info
            
    gain = {}
    for g in entropyattr:
        gain[g] = classentropy - entropyattr[g]
    
    node = max(gain, key=gain.get)
    tree = insertnode(tree, addto, node)
    
    for nd in attr[node]:
        tree = insertnode(tree, node, nd)
        newdata = data[data[node] == nd].drop(node, axis=1)
        new_attributelist = list(newdata)[:-1]  # Rename attributelist
        tree = getnextnode(newdata, new_attributelist, concept, conceptvals, tree, nd)  # Correct the typo
        
    return tree

def main():
    data = pd.read_csv('PlayTennis.csv')
    attributelist = list(data)[:-1]  # Rename atrributelist
    concept = str(list(data)[-1])
    conceptvals = list(set(data[concept]))
    tree = getnextnode(data, attributelist, concept, conceptvals, {'root': 'None'}, 'root')
    return tree

import pandas as pd
def main():
    data = pd.read_csv('PlayTennis.csv')
    attributelist = list(data)[:-1]
    concept =  str(list(data)[-1])
    conceptvals = list(set(data[concept]))
    tree = getnextnode(data,attributelist,concept,conceptvals,{'root' : 'None'},'root')
    return tree
tree = main()['root']
df=pd.read_csv('PlayTennis.csv')
def test(tree,d):
    for k in tree:
        for v in tree:
            if(d[k] == v and isinstance (tree[k][v],dict)):
                test(tree[k][v],d)
            elif(d[k]==v):
                print("classification:" +tree[k][v])

print(tree)
