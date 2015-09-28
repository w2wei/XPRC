def kskipngrams(sentence,k,n):
    "Assumes the sentence is already tokenized into a list"
    # modify this algorithm:
    # 1. if the sentence is shorter than k, automatically decrease the window size
    if n == 0 or len(sentence) == 0:
        return None
    grams = []
    for i in range(len(sentence)-n+1):
        grams.extend(initial_kskipngrams(sentence[i:],k,n))
    return grams

def initial_kskipngrams(sentence,k,n):
    if n == 1:
        return [[sentence[0]]]
    grams = []
    for j in range(min(k+1,len(sentence)-1)):
        kmjskipnm1grams = initial_kskipngrams(sentence[j+1:],k-j,n-1)
        if kmjskipnm1grams is not None:
            for gram in kmjskipnm1grams:
                grams.append([sentence[0]]+gram)
    return grams

if __name__=="__main__":
    sentence = "local environment factor duodenum may contribut differ molecular chang occur adenoma-to-carcinoma sequenc periampullari compar colon tumorigenesi".split(" ")
    skg = kskipngrams(sentence,5,2)
    print skg