from collections import defaultdict,Counter
import math
reviews=[("fun,couple,love,love","comedy"),
         ("fast,furious,shoot","action"),
         ("couple,fly,fast,fun,fun","comedy"),
         ("furious,shoot,shoot,fun","action"),
        ("fly,fast,shoot,love","action")]
D="fast,couple,shoot,fly"
def tokensize(text):
    return text.split(",")
class_docs=defaultdict(list)
vocabulary=set()
class_count=defaultdict(int)
for review,category in reviews:
    tokens=tokensize(review)
    class_docs[category].extend(tokens)
    class_count[category]+=1
    vocabulary.update(tokens)
vocab_size=len(vocabulary)
total_docs=len(reviews)
priors={category:count/total_docs for category,count in class_count.items()}
likelihoods={}
for category,tokens in class_docs.items():
    token_counts=Counter(tokens)
    total_words=len(tokens)
    likelihoods[category]={word:(token_counts[word]+1)/(total_words+vocab_size) for word in vocabulary }
tokens=tokensize(D)
posteriors={}
for category in priors:
    log_prob=(priors[category])
    for token in tokens:
        log_prob*=(likelihoods[category].get(token,1/(len(class_docs[category])+vocab_size)))
    posteriors[category]=log_prob
most_likely_class=max(posteriors,key=posteriors.get)
print('Posterior Probability:',posteriors)
print(f"The most likely class for the document '{D}'is:{most_likely_class}")
