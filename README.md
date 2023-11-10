**Stock prediction**is crucial for quantitative analysts and investment companies. Stocks’ trends,
however, are affected by a lot of factors such as interest rates, inflation rates and financial news [12].
To predict stock prices accurately, one must use these variable information. In particular, in the
banking industry and financial services, analysts’ armies are dedicated to pouring over, analyzing,
and attempting to quantify qualitative data from news. A large amount of stock trend information is
extracted from the large amount of text and quantitative information that is involved in the analysis.
Investors may judge on the basis of technical analysis, such as charts of a company, market indices,
and on textual information such as news blogs or newspapers. It is however difficult for investors
to analyze and predict market trends according to all of these information [22]. A lot of artificial
intelligence approaches have been investigated to automatically predict those trends [3]. For instance,
investment simulation analysis with artificial markets or stock trend analysis with lexical cohesion
based metric of financial news’ sentiment polarity. Quantitative analysis today is heavily dependent
on data. However, the majority of such data is unstructured text that comes from sources like financial
news articles. The challenge is not only the amount of data that are involved, but also the kind
of language that is used in them to express sentiments, which means emoticons. Sifting through
huge volumes of this text data is difficult as well as time-consuming. It also requires a great deal of
resources and expertise to analyze all of that [4].
To solve the above problem, in this paper we use **sentiment analysis** to extract information from
textual information. Sentiment analysis is the automated process of understanding an opinion about
a given subject from news articles [5]. The analyzed data quantifies reactions or sentiments of
the general public toward people, ideas or certain products and reveal the information’s contextual
33rd Conference on Neural Information Processing Systems
polarity. Sentiment analysis allows us to understand if newspapers are talking positively or negatively
about the financial market, get key insights about the stock’s future trend market.
We use valence aware dictionary and sentiment reasoner (VADER) to extract sentiment scores.
VADER is a lexicon and rule-based sentiment analysis tool attuned to sentiments that are expressed
in social media specifically [6]. VADER has been found to be quite successful when dealing with NY
Times editorials and social media texts. This is because VADER not only tells about the negativity
score and positively but also tells us about how positive or negative a sentiment is.
However, news reports are not all objective. We may increase bias because of some non-objective
reports, if we rely on the information that is extracted from the news for prediction fully. Therefore,
in order to enhance the prediction model’s robustness, we will adopt differential privacy (DP) method.
DP is a system for sharing information about a dataset publicly by describing groups’ patterns within
the dataset while withholding information about individuals in the dataset. DP can be achieved if the
we are willing to add random noise to the result. For example, rather than simply reporting the sum,
we can inject noise from a Laplace or gaussian distribution, producing a result that’s not quite exact,
that masks the contents of any given row.
In the last several years a promising approach to private data analysis has emerged, based on DP,
which ensures that an analysis outcome is "roughly as likely" to occur independent of whether any
individual opts in to, or to opts out of, the database. In consequence, any one individual’s specific data
can never greatly affect the results. General techniques for ensuring DP have now been proposed, and
a lot of datamining tasks can be carried out in a DP method, frequently with very accurate results [21].
We proposed a DP-LSTM neural network, which increase the accuracy of prediction and robustness
of model at the same time.
The remainder of the paper is organized as follows. In Section 2, we introduce stock price model, the
sentiment analysis and differential privacy method. In Section 3, we develop the different privacyinspired LSTM (DP-LSTM) deep neural network and present the training details. Prediction results
are provided in Section 4. Section 5 concludes the paper.
**2 Problem Statement**
In this section, we first introduce the background of the stock price model, which is based on the
autoregressive moving average (ARMA) model. Then, we present the sentiment analysis details of
the financial news and introduce how to use them to improve prediction performance. At last, we
introduce the differential privacy framework and the loss function.
**2.1 ARMA Model**
The ARMA model, which is one of the most widely used linear models in time series prediction [17],
where the future value is assumed as a linear combination of the past errors and past values. ARMA
is used to set the stock midterm prediction problem up. Let XA
t be the variable based on ARMA at
time t, then we have
XA
t = f1({Xt−i}
p
i=1) = µ +
Xp
i=1
φiXt−i −
Xq
i=1
ψj t−j + t, (1)
where Xt−i denotes the past value at time t − i; t denotes the random error at time t; φi and ψj are
the coefficients; µ is a constant; p and q are integers that are often referred to as autoregressive and
moving average polynomials, respectively.
**2.2 Sentiment Analysis**
Another variable highly related to stock price is the textual information from news, whose changes
may be a precursor to price changes. In our paper, news refers to a news article’s title on a given trading
day. It has been used to infer whether an event had informational content and whether investors’
interpretations of the information were positive, negative or neutral. We hence use sentiment analysis
to identify and extract opinions within a given text. Sentiment analysis aims at gauging the attitude,
sentiments, evaluations and emotions of a speaker or writer based on subjectivity’s computational
treatment in a text [19]-[20].
To take the sentiment analysis results of the financial news into account, we introduce the sentimentARMA model as follows
Xˆ
t = αXA
t + λSA
t + c = αXA
t + λf2({St−i}
p
i=1
| {z }
Sentiment
) + c, (2)
where α and λ are weighting factors; c is a constant; and f2(·) is similar to f1(·) in the ARMA model
(1) and is used to describe the prediction problem.
In this paper, the LSTM neural network is used to predict the stock price, the input data is the previous
stock price and the sentiment analysis results. Hence, the sentiment based LSTM neural networknsidering news factors, because news can’t guarantee
full notarization and objectivity, many times extreme news will have a big impact on prediction
models. To solve this problem, we consider entering the idea of the differential privacy when training.
In this section, our DP-LSTM deep neural network training strategy is presented. The input data
consists of three components: stock price, sentiment analysis compound score and noise.
3.1 Data Preprocessing and Normalization
3.1.1 Data Preprocessing
The data for this project are two parts, the first part is the historical S&P 500 component stocks,
which are downloaded from the Yahoo Finance. We use the data over the period of from 12/07/2017
to 06/01/2018. The second part is the news article from financial domain are collected with the same
time period as stock data. Since our paper illustrates the relationship between the sentiment of the
news articles and stocks’ price. Hence, only news article from financial domain are collected. The
data is mainly taken from Webhose archived data, which consists of 306242 news articles present in
JSON format, dating from December 2017 up to end of June 2018. The former 85% of the dataset is
used as the training data and the remainder 15% is used as the testing data. The News publishers for
this data are CNBC.com, Reuters.com, WSJ.com, Fortune.com. The Wall Street Journal is one of
the largest newspapers in the United States, which coverage of breaking news and current headlines
from the US and around the world include top stories, photos, videos, detailed analysis and in-depth
thoughts; CNBC primarily carries business day coverage of U.S. and international financial markets,
which following the end of the business day and on non-trading days; Fortune is an American
multinational business magazine; Reuters is an international news organization. We preprocess the
raw article body and use NLTK sentiment package alence Aware Dictionary and Sentiment Reasoner
(VADER) to extract sentiment scores.
The stocks with missing data are deleted, and the dataset we used eventually contains 451 stocks and
4 news resources (CNBC.com, Reuters.com, WSJ.comFortune.com.). Each stock records the adjust
close price and news compound scores of 121 trading days. Training input: 92 days
Training output: 92 days
12/07/2017 04/20/2018 05/04/2018 05/18/2018
12/08/2017 04/23/2018 05/07/2018 05/21/2018
Testing output: 9 days
Testing input: 9 days
Figure 4: Schematic diagram of rolling window.
A rolling window with size 10 is used to separate data, that is, We predict the stock price of the next
trading day based on historical data from the previous 10 days, hence resulting in a point-by-point
prediction [15]. In particular, the training window is initialized with all real training data. Then we
shift the window and add the next real point to the last point of training window to predict the next
point and so forth. Then, according to the length of the window, the training data is divided into 92
sets of training input data (each set length 10) and training output data (each set length 1). The testing
data is divided into input and output data of 9 windows (see Figure 4).
5
3.1.2 Normalization
To detect stock price pattern, it is necessary to normalize the stock price data. Since the LSTM neural
network requires the stock patterns during training, we use “min-max” normalization method to
reform dataset, which keeps the pattern of the data [11], as follow:
Xn
t =
Xt − min(Xt)
max(Xt) − min(Xt)
, (10)
where Xn
t denotes the data after normalization. Accordingly, de-normalization is required at the end
of the prediction process to get the original price, which is given by
Xˆ
t = Xˆ n
t
[max(Xt) − min(Xt)] + min(Xt), (11)
where Xˆ n
t denotes the predicted data and Xˆ
t denotes the predicted data after de-normalization.
Note that compound score is not normalized, since the compound score range from -1 to 1, which
means all the compound score data has the same scale, so it is not require the normalization processing.
3.2 Adding Noise
We consider the differential privacy as a method to improve the robustness of the LSTM predictions [8].
We explore the interplay between machine learning and differential privacy, and found that differential
privacy has several properties that make it particularly useful in application such as robustness
to extract textual information [9]. The robustness of textual information means that accuracy is
guaranteed to be unaffected by certain false information [10].
The input data of the model has 5 dimensions, which are the stock price and four compound scores
as (Xt
, St
1
, St
2
, St
3
, St
4
), t = 1, ..., T, where Xt
represents the stock price and S
t
i
, i = 1, ..., 4
respectively denote the mean compound score calculated from WSJ, CNBC, Fortune and Reuters.
According to the process of differential privacy, we add Gaussian noise with different variances to the
news according to the variance of the news, i.e., the news compound score after adding noise is given
by
Set
i = S
t
i + N (0, λvar(Si)), i = 1, ..., 4, (12)
where var(·) is the variance operator, λ is a weighting factor and N (·) denotes the random Gaussian
process with zero mean and variance λvar(Si).
We used python to crawl the news from the four sources of each trading day, perform sentiment
analysis on the title of the news, and get the compound score. After splitting the data into training sets
and test sets, we separately add noise to each of four news sources of the training set, then, for n-th
stock, four sets of noise-added data (Xn
t
, Set
1
, St
2
, St
3
, St
4
), (Xn
t
, St
1
, Set
2
, St
3
, St
4
), (Xn
t
, St
1
, St
2
, Set
3
, St
4
),
(Xn
t
, St
1
, St
2
, St
3
, Set
4
) are combined into a new training data through a rolling window. The stock price
is then combined with the new compound score training data as input data for our DP-LSTM neural
network.
3.3 Training Setting
The LSTM model in figure 3 has six layers, followed by an LSTM layer, a dropout layer, an LSTM
layer, an LSTM layer, a dropout layer, a dense layer, respectively. The dropout layers (with dropout
rate 0.2) prevent the network from overfitting. The dense layer is used to reshape the output. Since
a network will be difficult to train if it contains a large number of LSTM layers [16], we use three
LSTM layers here.
In each LSTM layer, the loss function is the mean square error (MSE), which is the sum of the
squared distances between our target variable and the predicted value. In addition, the ADAM [17] is
used as optimizer, since it is straightforward to implement, computationally efficient and well suited
for problems with large data set and parameters.
There are many methods and algorithms to implement sentiment analysis systems. In this paper,
we use rule-based systems that perform sentiment analysis based on a set of manually crafted rules.
Usually, rule-based approaches define a set of rules in some kind of scripting language that identify
subjectivity, polarity, or the subject of an opinion. We use VADER, a simple rule-based model for
general sentiment analysis.
**Conclusion**
In this paper, we integrated the deep neural network with the famous NLP models (VADER) to identify
and extract opinions within a given text, combining the stock adjust close price and compound score to
reduce the investment risk. We first proposed a sentiment-ARMA model to represent the stock price,
which incorporates influential variables (price and news) based on the ARMA model. Then, a DPLSTM deep neural network was proposed to predict stock price according to the sentiment-ARMA
model, which combines the LSTM, compound score of news articles and differential privacy method.
News are not all objective. If we rely on the information extracted from the news for prediction fully,
we may increase bias because of some non-objective reports. Therefore, the DP-LSTM enhance
robustness of the prediction model. Experiment results based on the S&P 500 stocks show that
the proposed DP-LSTM network can predict the stock price accurately with robust performance,
especially for S&P 500 index that reflects the general trend of the market. S&P 500 prediction results
show that the differential privacy method can significantly improve the robustness and accuracy
