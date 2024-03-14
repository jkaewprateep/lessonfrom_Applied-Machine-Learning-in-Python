# University of Michigan - Applied Machine Learning in Python - notes
University of Michigan - Applied Machine Learning in Python - notes

## 🧸💬 What are prediction scores and why are the provided prediction scores not equal to the summation of the error square of scores and prediction?
💃( 👩‍🏫 )💬 We need representing values because it is not a measurement but a representation of the same method and from the expecting method, communication values are not correct values in mathematics but it is meaning in most cases. </br>
👧💬 🎈 The first scores function values in the green graph 🟢 of the second couple graph is the addition of scores without weights and the orange graph 🟠 of the third graph is distribution weights. </br>
🐐💬 Yes, it is but how do we compare the prediction scores directly with the evaluation scores⁉️ She intends to tell you it is a prediction score from a function, not actual values⁉️ </br>
🦤💬 That is because often ```mistakes created when directly compared to prediction scores from different problem statements``` that is because they may lead to similar closed values but from the different domains, we try to improve both accuracy and precision when the number is stepping closer relations. </br>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/01.png">
</p>

## 🧸💬 Why periodic function steps help with prediction location and tricks.
🦭💬 There are natural axes which are the expectation axis and common features we need setup a cort that creates the most possibility to bring our zero to the target. </br>
🐯💬 It is a visual eye for near and far side expectations, and it does not require actual scales when it is power scales. </br>
🐑💬 ➰ From the graph there are potential points in ranges and moving they are changing in location related to the source of the observation, remarks of labels. </br>
🐨🎁🎵🎶 This is called ```specialized effects``` because technology is built from our expected domain and working environment, newly discovered or secrete tabs have more effects against specialist effects when specialists are required to build manageable technologies. </br>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/03.png">
</p>

## 🧸💬 What are quadratic functions and what are coefficients order and priority?
💃( 👩‍🏫 )💬 Prediction for accelerated actions from known databases use SGD because it is prediction, clarifies, and studies well as information and study angles are from our solution and statements. </br>
🦤💬 To succeed we cannot directive the action wind strikes and collision domain to work on the true object, we had this small capability in our calculation called work resistances as coefficients of the quadratic functions. ( The picture is a simulation and it could happen fast or as long as manufacturing designs ) </br>
👧💬 🎈 Most effective is not the most variables but it is the most precise target as objective required because pieces of equipment are controlled by humans. </br>
🐐💬 They are challenging on fast responsive feedback systems and automatically work when sometimes the most stable ( dull ) perform the perfect task on this. </br>

<p align="center" width="100%">
    <img width="75%" src="https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/04.png">
</p>

## 🧸💬 How does recall effects learning rates and output result from control function?
🐨🎁🎵🎶 A percentage success is not the highest accuracy estimation number but effective of costs and responsible functions with correct degree calculation. </br>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/05.png">
</p>

## 🧸💬 Precision recall curve and estimates value by graph linearity for example prediction output of the function for value of specific control values input.
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/06.png?raw=true "Title")

## 🧸💬 Heats mapping of liner distribution, global linear distribution display of weight, and labels matrix ( GridSearch ).
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/07.png?raw=true "Title")

## 🐑💬 ➰ Regression problem with regression models ##
🐑💬 ➰ This is not fool or tricks you but it is how we deal with the problem with linear-regression model and linear regression method, you may apply both SGD momentums. logistics and Adam.
```
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, Normalizer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
import pickle
from os.path import exists

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from random import randrange

def engagement_model():
    rec = None
    df_test = pd.read_csv('assets/test.csv')
    df_train = pd.read_csv('assets/train.csv')
    
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    ###
    # df_train = df_train[df_train["normalization_rate"] > 0]
    df_train["avg_speakerspeed"] = np.nanmean(df_train["speaker_speed"]) - df_train["speaker_speed"]
    df_train["avg_easiness"] = np.nanmean(df_train["easiness"]) - df_train["easiness"]
    df_train["avg_normalization_rate"] = np.nanmean(df_train["normalization_rate"]) - df_train["normalization_rate"]

    df_test["avg_speakerspeed"] = np.nanmean(df_test["speaker_speed"]) - df_test["speaker_speed"]
    df_test["avg_easiness"] = np.nanmean(df_test["easiness"]) - df_test["easiness"]
    df_test["avg_normalization_rate"] = np.nanmean(df_test["normalization_rate"]) - df_test["normalization_rate"]
    ###
    
    # For prediction is engagement == True;
    df_train_X = df_train[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
                           "speaker_speed", "silent_period_rate"]];
    
    
    df_train_y = df_train[["engagement"]];
    df_train_y["engagement"] = df_train_y["engagement"].apply( lambda x : 1.0 if x == True else 0.0 );

    X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_train_y, random_state = 0)
    
    scaler = StandardScaler();
    # scaler = StandardScaler().fit(df_train_X); #<<<<<<<<<<<<<<<<<<<<
    
    X_train_scaled = scaler.fit_transform(df_train_X)
    X_test_scaled = scaler.transform(df_train_X)
    
    #########################################################
    load_model = MLPClassifier(hidden_layer_sizes=[152, 14], activation='tanh', solver='adam', alpha=0.01, batch_size=1024, learning_rate='adaptive', 
                               learning_rate_init=0.00000000001, power_t=0.5, max_iter=300, shuffle=True, random_state=1, tol=0.0001, verbose=False, 
                               warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                               epsilon=1e-09, n_iter_no_change=10, max_fun=15000)     
    #########################################################
    
    ###
    filename = 'trained_model.sav'
    if exists(filename):
        load_model = pickle.load(open(filename, 'rb'));
        scaler = pickle.load(open('Scaler.pk', 'rb'));

    else:    
        
        load_model = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(128, 256, 256, 256, 128, 32), random_state=1,
                                       alpha=0.01, batch_size=1024, learning_rate='adaptive', power_t=0.5, shuffle=True,
                                       max_iter=300,verbose=10,learning_rate_init=0.00001, momentum=0.9, nesterovs_momentum=True, 
                                       early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-09, 
                                       n_iter_no_change=10, max_fun=15000);
        
        for i in range(1000):
            ###
            df_train = pd.read_csv('assets/train.csv')
            random_state = randrange(42);
            df_train = shuffle(df_train, random_state=random_state);
            df_train_X = df_train[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
                           "speaker_speed", "silent_period_rate"]];
            df_train_y = df_train[["engagement"]];
            df_train_y["engagement"] = df_train_y["engagement"].apply( lambda x : 1.0 if x == True else 0.0 );
            
            X_train_scaled = scaler.fit_transform(df_train_X)
            
            
            load_model = load_model.fit(X_train_scaled, df_train_y);
            ###

            pickle.dump(load_model, open(filename, 'wb'));
            pickle.dump(scaler, open('Scaler.pk', 'wb'));
            ##########
            load_model = pickle.load(open(filename, 'rb'));
            scaler = pickle.load(open('Scaler.pk', 'rb'));

            print( i, load_model.score(X_train_scaled, df_train_y) );

            if load_model.score(X_train_scaled, df_train_y) > 0.95 :
                print( "load_model.score(X_train_scaled, df_train_y) > 0.95" );
                break;
            ###
        
        pickle.dump(load_model, open(filename, 'wb'));
        pickle.dump(scaler, open('Scaler.pk', 'wb'));

    
    df_testrec = df_test[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
                           "speaker_speed", "silent_period_rate"]];
    
    to_predictdf = scaler.transform(df_testrec)

    predictions = load_model.predict(to_predictdf)
    
    index_integer = pd.DataFrame([]);
    index_integer["id"] = df_test["id"].astype("int");
    index_integer["engagement"] = index_integer.apply( lambda x : predictions[x.index] );
    
    predictions = index_integer;
    predictions = predictions.set_index("id");

    
    print('Breast cancer dataset')
    print('Accuracy of NN classifier on training set: {:.2f}'
    .format(load_model.score(X_train_scaled, df_train_y)))
    print('Accuracy of NN classifier on test set: {:.2f}'
    .format(load_model.score(X_test_scaled, df_train_y)))
    
    return predictions.iloc[:,0];
```
