import logging

import gensim.downloader
w2v = gensim.downloader.load('word2vec-google-news-300')

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def analogy(a,b,c):
    print(a+"  :  "+b+"  ::  "+c+"  :  ?")
    print([(w,round(c,3)) for w,c in w2v.most_similar(positive=[c,b], negative= [a])])
    print("\n")

if __name__ == "__main__":

    analogy('man', 'king', 'woman')

    logging.info("Stereotype about gender roles in medicine")
    analogy('man', 'doctor', 'woman')
    analogy('woman', 'doctor', 'man')

    logging.info("Stereotype about gender roles in victimhood")
    analogy('man', 'victim', 'woman')
    analogy('woman', 'victim', 'man')

    analogy('Paris', 'France', 'Tokyo')
    analogy('morning', 'breakfast', 'evening')
    analogy('walking', 'walk', 'swimming')

    #====================

    analogy('teacher', 'school', 'doctor')
    analogy('sun', 'daytime', 'moon')
    analogy('Japan', 'sushi', 'Italy')

    #====================
    analogy('man', 'computer_programmer', 'woman')
    analogy('woman', 'computer_programmer', 'man')

    #=====================
    analogy('Christian', 'good', 'Muslim')
    analogy('Muslim', 'good', 'Christian')


