# Unsupervised Content-based Image Retrieval on SOFM

## 待辦
- [ ] feature取法
- [ ] 動機修改 (google圖片搜尋很強大....balabala)
- [ ] 結論 (unsupervised image retrival很難，training 很花時間等等)
- [ ] usage (如何使用我們的code)
- [ ] code中各function代表的意義
- [ ] (可選) 心得
- [ ] dataset
- [ ] eigen?

## 動機
隨著機器學習技術的成熟，圖片的辨識與分類變得更加容易，但大多數的辨識還是必須仰賴Supervised Neural Network，工程師必須不斷的告訴電腦這樣的答案對或錯，提供NN更新權重的標準；但資料庫越來越大，Supervised的方式不僅效率低落，也浪費人力，因此我們希望能利用Unsupervised的方式來進行圖像的處理。
我們希望類神經網路能直接依據圖片本身的內容作出分類，取代人力一張圖片一張圖片進行tag的過程，以Google的圖片搜尋為例，我們可以直接輸入一張新的圖片(Google資料庫內所沒有的，)，
在上半學期，老師介紹了許多 unsupervised neural network，一開始我們對 Hopfield NN 感到十分新奇，想利用HNN的特性來實作 Image Retrieval，但經過老師建議後，改為使用SOFM進行 Unsupervised Image Retrieval，並搭配不同Feature Extraction 方法，嘗試其效果。 


## Self Organized Feature Mapping(SOFM)
 * 原理
  SOFM的概念是大腦在處理資訊時，處理相同資訊的神經會聚集在一起，因此會形成一個一個聚落(Cluster)，而其延伸在資料處理時，我們可以將N維的資料映射到一個二維平面，進行降維，再觀察聚集在一起的資料，是否有相同特性。
  進行SOFM的流程如下：
    1. 對每個node的權重進行初始化
    2. 從training data中隨機選一筆資料
    3. 找出一個Node(BMU, Best Matching Unit)，其與這筆資料之間有著最小的Euclidean Distance
     $$Dist = \sqrt{\sum^{i=n}_{i=0}(V_i-W_i)^2}$$
     $V$為該筆training data
     $W$為node之權重
    4. 計算更新半徑：
$$\sigma(t) = \sigma_0exp(-\frac{t}{\lambda})$$
        $\sigma_0$為mapping的大小，在 $t = 0$ 時，更新半徑會涵蓋整個map
        $\lambda$ 為 time constant
    5. 調整該半徑內所有node的權重，使其更接近input vector，越靠近BMU的node其調整幅度更大，更新公式為：
$$W(t+1) = W(t) + \Theta(t)L(t)(V(t)-W(t))$$
$$L(t) = L_0exp(-\frac{t}{\lambda})$$
$$\Theta(t) = exp(-\frac{dist^2}{2\sigma^2(t)})\qquad t = 1,2,3,....$$
$L(t)$ 為Learning rate，會隨著時間而降低
$\Theta(t)$ 為隨著與BMU之間的Euclidean Distance改變，其更新幅度的大小，如下圖
![](http://www.ai-junkie.com/ann/som/images/graph1.jpg)

    6. 重複 2. 直到跑完所有epoch
    7. 每個node會將離自己最近的64張training data存入自身
    8. 比較每個node，若任兩個Node中，有50個以上相同的data，則這兩個Node會被分類到同一個cluster，依照上述過程，將所有Node進行分類
    9. 輸入test data，找出離他最近的node，輸出該node所屬的cluster中所有的圖片

* 實作
  我們試圖將SOFM以視覺化的方式表現出來，除了說明其降維的功效外，也能觀察其將node進行分類並依據種類而聚集的狀態。
  首先設定map為 500 x 500 個 node，每個node的weight皆為隨機的三維變數，接著使用我們事先寫好的training data(0到1之間)進行training。
 train完後將node的weight放大成0到255間，並以RGB的方式，使每一個Node代表一個Pixel，將所有Node輸出成一張pixel 500 x 500的圖片，如下圖：從左而右分別為4筆、10筆、100筆input data。
![](https://i.imgur.com/tHaX3GB.jpg =100x100) ![](https://i.imgur.com/NdHA6vO.jpg =100x100)  ![](https://i.imgur.com/e2PuNCh.jpg =100x100)

- 複雜度
	依此設計來講時間複雜度為 $O(nf)$
	$n$為node的數量，$f$為feature的數量
	下面的設計中node數皆為400
	feature數則盡量接近1000
## Feature Extraction之方法與結果

### Color Histogram 
 * 原理 
 Color Histogram是在許多Content Based Image Retrieval System中被廣泛採用的feature。它所描述的是不同色彩在整幅圖像中所佔的比例，但是對於每一個顏色在該圖片的相對位置卻不夠重視。Color Histogram特別適於描述那些難以進行自動分割的圖像。
 * 方法
 先將圖片轉成一個三維的資訊，分別為長、寬、以及RGB，然後對每一格根據RGB的值算出一個0~255之間的值，即可求出Color Histogram。
 * 結果
 前方大小為120x80的原圖，後方為經過我們系統得到的最相近的64張圖
 ![](https://i.imgur.com/KD1p9rc.jpg =120x80)![](https://i.imgur.com/a1GKDDo.png =400x400)
![](https://i.imgur.com/1xCc46S.jpg)![](https://i.imgur.com/gYXrUte.png =400x400)
![](https://i.imgur.com/0RTP0zi.jpg)![](https://i.imgur.com/Bh6eXTg.png =400x400)
 * 討論
由以上的結果可以看到，把histogram作為主要的feature取法是不夠具有代表性的，因為只考慮到顏色0~255的數值且無法有效的以此作為判斷圖片內物體的依據，所以我們改尋找其他能夠有如圖片內物體的edge的feature取法。

### Speeded Up Robust Features (SURF)
 SURF是一種圖像辨識與描述的算法，他使用了海森矩陣的行列式值作特徵點偵測並用積分圖加速運算；SURF 的描述子基於 2D 離散小波變換 響應並且有效地利用了積分圖。在computer vision的領域中，可用於物件辨識與3D構件，如下圖，他可以找出圖片的特徵點，當圖片被扭曲時，該特徵點依舊能保持原有的特性，從而將兩張圖片進行比對。
 ![](https://i.imgur.com/em6ma5w.png)
 ![](https://i.imgur.com/xRZJpEX.png)

 也因為SURF能夠辨認出圖片當中的特徵點，而Opencv有支援SURF的運算，因此我們想利用這些特徵點當作圖片的feature，每張圖片取16個特徵點，每個特徵點為64維，再將他放進SOFM的架構內進行training。
 * 結果與討論
 當我們仔細去檢視每個Node當中儲存的圖片，發現圖片幾乎是呈現隨機分布，沒有辦法用人眼判讀其分類是否正確，推究原因後，我們認為我們對SURF所取出的特徵點不夠了解，無法判斷每個特徵點是否能代表其物件，但我們時間不足，也只能繼續尋找下一個方法。
 
### Brutal
因為SURF的feature無法使用SOFM的方式來做分類，因此我們又另外找方法。這次我們想要直接拿圖片的每個pixel來做為feature，而一個pixel又有RGB三個資訊，因此總共有 80 * 120 * 3 = 28800個feature。以我們的硬體設備是沒有辦法跑的，因此必須要降維。我們嘗試過PCA和autoencoder兩種方式，最後採用autoencoder，不過兩種降維方式拿來跑SOFM各自產生了不一樣的問題


### PCA

在多元統計分析中，PCA是一種降低資料量的技術。主要是用於在feature降維的時候還是能夠保持feature的代表性。這是通過保留低階主成分，忽略高階主成分做到的。這樣低階成分往往能夠保留住數據的最重要方面，PCA不只可用於電腦視覺辨識，亦可用於語音辨識或其他機器學習領域。

- 作法和結果
透過呼叫scikit-learn裏頭的PCA套件來將28800維的feature降為1024維，並發現結果卻不如預期，下方為透過PCA把feature降維做出來的結果，前方為原圖、後方為透過我們系統拿出的相近的圖，

![](https://i.imgur.com/4udT1wJ.jpg)![](https://i.imgur.com/JCZWuLZ.jpg =400x400)
![](https://i.imgur.com/35JZpul.jpg)![](https://i.imgur.com/xaAO6TY.png =400x400)
![](https://i.imgur.com/OoROeKM.jpg)![](https://i.imgur.com/cgtKTzH.png =400x400)







### Autoencoder
使用autoencoder的原因是因為比起PCA，autoencoder的降維方式對二維的資料比較有用(畢竟也是用CNN達成)。因此在教授的建議下，我們改使用autoencoder。
首先參考了這個網頁有關使用keras來實作的教學：[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)。不過他使用的data是MNIST，是黑白的，而我們要encoder的對象是彩色的，因此某些參數可能需要修改，我們最後用了以下的model當作我們的encoder。


```python
	ACT = 'tanh'
	PADDING = 'same'
	input_img = Input(shape=(80, 120, 3))  # adapt this if using `channels_first` image data format
	x = Conv2D(64, (3, 3), activation=ACT, padding=PADDING)(input_img)
	x = MaxPooling2D((2, 2), padding=PADDING)(x)
	x = Conv2D(32, (3, 3), activation=ACT, padding=PADDING)(x)
	x = MaxPooling2D((2, 2), padding=PADDING)(x)
	x = Conv2D(8, (3, 3), activation=ACT, padding=PADDING)(x)
	encoded = MaxPooling2D((2, 2), padding=PADDING)(x)
	encoder = Model(input_img, encoded)


	x = Conv2D(8, (3, 3), activation=ACT, padding=PADDING)(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation=ACT, padding=PADDING)(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation=ACT, padding=PADDING)(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding=PADDING)(x)
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='sgd', loss='mse')
```
值得一提的是使用NN做autoencoder的時候CNN的padding要使用same，否則CNN會讓資料數量下降，讓我們encode之後無法decode回來。
另外，之所以要train autoencoder而不是只train encoder就好，是為了看我們的model好不好。而在改了幾次model的架構之後，有了不錯的結果
![](https://i.imgur.com/tPMvNjv.png)
上方的表格中，前三張圖片是不在training資料裡面的，因此可以確定我們train的model是有用的，可以在一定的程度下把圖片encode再decode回相似的原圖


#### 結果
利用autoencoder我們把原本整張圖的80\*120\*3 = 28800 個 features 降到 10\*15\*8 = 1200個，有效強化了train的效率。然而實作出來的結果甚至比PCA還要糟糕。
我們似乎與做HNN那組一樣，出現了fixpoint的問題。第一次輸出圖片之後發現所有的node幾乎都在圖片是盤子的附近，也就是說100個node全部都是一個cluster，不管丟什麼圖片test都只會跑出盤子。而把所有的盤子刪掉之後，fixpoint變成門的圖片。改了不少參數之後仍然有這個問題存在，因此我們推測encode之後的feature會讓我們的sofm算法的node特別趨近於某些圖片，無法把這寫node分開。
 * 討論


## 程式執行環境與指令
 * 環境：Python 3.6
 * 指令：python3 final.py <command>
* Dataset：
* command
  auto_train：將
    

## 結論
從結果來說我們的做法算是失敗了，