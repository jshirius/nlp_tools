# coding: UTF-8

import matplotlib.pyplot as plt
from wordcloud import WordCloud

#wordcloudを作成する
def create_wordcloud(text:list , image_path:str):
    """wordcloudを作成する
    
    Args:
        text (list): テキストのリスト
        image_path (str): imageのpath
    """
    # 環境に合わせてフォントのパスを指定する。
    fpath = "./ipamp.ttf"
    text = " ". join(text)
    # ストップワードの設定
    stop_words = [ u'てる', u'いる', u'なる', u'れる', u'する', u'ある', u'こと', u'これ', u'さん', u'して', \
             u'くれる', u'やる', u'くださる', u'そう', u'せる', u'した',  u'思う',  \
             u'それ', u'ここ', u'ちゃん', u'くん', u'', u'て',u'に',u'を',u'は',u'の', u'が', u'と', u'た', u'し', u'で',u'ます',u'です' \
             u'ない', u'も', u'な', u'い', u'か', u'ので', u'よう', u'',"http","https","RT"]

    wordcloud = WordCloud(background_color="black",font_path=fpath, width=900, height=500, \
                          stopwords=set(stop_words)).generate(text)

    plt.figure(figsize=(15,12))
    plt.imshow(wordcloud)
    plt.axis("off")
  

    # save as png
    plt.savefig(image_path) # -----(2)
    plt.show()