# University of Michigan - Applied Machine Learning in Python - notes
University of Michigan - Applied Machine Learning in Python - notes

## 🧸💬 What are prediction scores and why are the provided prediction scores not equal to the summation of the error square of scores and prediction?
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/01.png?raw=true "Title")

## 🧸💬 Why periodic function steps help with prediction location and tricks.
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/03.png?raw=true "Title")

## 🧸💬 What are quadratic functions and what are coefficients order and priority?
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/04.png?raw=true "Title")

## 🧸💬 How does recall effects learning rates and output result from control function?
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/05.png?raw=true "Title")

## 🧸💬 Precision recall curve and estimates value by graph linearity for example prediction output of the function for value of specific control values input.
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/06.png?raw=true "Title")

## 🧸💬 Heats mapping of liner distribution, global linear distribution display of weight, and labels matrix ( GridSearch ).
![Alt text](https://github.com/jkaewprateep/lessonfrom_Applied-Machine-Learning-in-Python/blob/main/07.png?raw=true "Title")

## Regression problem with regression models ##
```
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, Normalizer
# from sklearn.model_selection import train_test_split
# # from sklearn.metrics import roc_auc_score
# import pickle
# from os.path import exists

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.utils import shuffle
# from random import randrange

# ###
# # !wget -O download.zip 'https://github.com/jkaewprateep/Applied-Machine-Learning-in-Python/archive/refs/heads/main.zip'
# # !unzip -o download.zip -d .
# # !cp ./Applied-Machine-Learning-in-Python-main/trained_model.zip ./trained_model.zip
# # !unzip -o trained_model.zip
# ###
# # !cp ./test.csv ./assets/test.csv
# # !cp ./train.csv ./assets/train.csv
# ###

# def engagement_model():
#     rec = None
#     df_test = pd.read_csv('assets/test.csv')
#     df_train = pd.read_csv('assets/train.csv')
    
#     # YOUR CODE HERE
#     # raise NotImplementedError()
    
#     ###
#     # df_train = df_train[df_train["normalization_rate"] > 0]
#     df_train["avg_speakerspeed"] = np.nanmean(df_train["speaker_speed"]) - df_train["speaker_speed"]
#     df_train["avg_easiness"] = np.nanmean(df_train["easiness"]) - df_train["easiness"]
#     df_train["avg_normalization_rate"] = np.nanmean(df_train["normalization_rate"]) - df_train["normalization_rate"]

#     df_test["avg_speakerspeed"] = np.nanmean(df_test["speaker_speed"]) - df_test["speaker_speed"]
#     df_test["avg_easiness"] = np.nanmean(df_test["easiness"]) - df_test["easiness"]
#     df_test["avg_normalization_rate"] = np.nanmean(df_test["normalization_rate"]) - df_test["normalization_rate"]
#     ###
    
#     # For prediction is engagement == True;
#     df_train_X = df_train[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
#                            "speaker_speed", "silent_period_rate"]];
    
#     # df_train_X = df_train[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
#     #                        "speaker_speed", "silent_period_rate", "avg_speakerspeed", "avg_easiness", "avg_normalization_rate"]];
    
#     # df_train_X = df_train[["easiness", "fraction_stopword_presence", "normalization_rate", 
#     #                        "speaker_speed", "silent_period_rate"]];
    
#     ###
#     # df_train_X["avg_speakerspeed"] = df_train_X["speaker_speed"].mean() - df_train_X["speaker_speed"]
#     ###
    
    
#     df_train_y = df_train[["engagement"]];
#     df_train_y["engagement"] = df_train_y["engagement"].apply( lambda x : 1.0 if x == True else 0.0 );
#     # df_train_y["engagement"] = df_train_y["engagement"].apply( lambda x : 1 if x == True else 0 );
#     # df_train_y["engagement"] = df_train_y["engagement"].apply( lambda x : 9 if x == True else 0 );
    
#     ###
#     # df_train_X_2 = df_test[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", "speaker_speed", "silent_period_rate"]];
#     # df_train_y_2 = df_test[["engagement"]];
#     # df_train_y_2["engagement"] = df_train_y_2["engagement"].apply( lambda x : 1 if x == True else 0 );
#     ###
    
#     # scaler = MinMaxScaler()
#     # scaler = PowerTransformer()
#     # scaler = StandardScaler();
#     # scaler = RobustScaler();
#     # scaler = QuantileTransformer();
#     # scaler = MaxAbsScaler();
#     # scaler = Normalizer();
#     # scaler = StandardScaler();

    
#     # scaler_2 = MinMaxScaler()
    

#     X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_train_y, random_state = 0)
    
#     ##########################
    
#     # X_train_scaled = scaler.fit_transform(X_train)
#     # X_test_scaled = scaler.transform(X_test)
    
#     scaler = StandardScaler();
#     # scaler = StandardScaler().fit(df_train_X); #<<<<<<<<<<<<<<<<<<<<
    
#     X_train_scaled = scaler.fit_transform(df_train_X)
#     X_test_scaled = scaler.transform(df_train_X)
    
#     ##########################
    
#     ###
#     # scaler_MM = QuantileTransformer();
#     # X_train_scaled = scaler_MM.fit_transform(X_train_scaled);
#     # X_test_scaled = scaler_MM.transform(X_test_scaled);
#     ###
    
    
    
#     ###
#     # X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_train_X_2, df_train_y_2, random_state = 0)
#     # X_train_scaled_2 = scaler.fit_transform(X_train_2)
#     # X_test_scaled_2 = scaler.transform(X_test_2)
#     ###
    
#     # load_model = MLPClassifier(hidden_layer_sizes = [2500, 400], alpha = 0.0001, learning_rate="adaptive", learning_rate_init=0.00001,
#     #        random_state = 0, solver='sgd');
    
#     # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#     # load_model = MLPClassifier(hidden_layer_sizes = [2048, 384], activation='tanh', solver='adam', alpha = 5.0, learning_rate="adaptive", learning_rate_init==0.0001, tol=1e-6, epsilon=1e-8,
#     #                            n_iter_no_change=15, early_stopping=True, random_state = 0, shuffle=True, beta_1=0.9, beta_2=0.98);
    
#     # alpha=0.0001
    
#     # load_model = MLPClassifier(solver='adam', activation='tanh',alpha=1e-4,hidden_layer_sizes=(50,50,50), random_state=1,max_iter=300,verbose=10,learning_rate_init=0.00001)
    
#     #########################################################
#     # load_model = MLPClassifier(hidden_layer_sizes=[128, 14], activation='tanh', solver='adam', alpha=5.0, batch_size=1024, learning_rate='adaptive', 
#     #                            learning_rate_init=0.00000000001, power_t=0.5, max_iter=300, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
#     #                            warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
#     #                            epsilon=1e-09, n_iter_no_change=10, max_fun=15000)
#     #########################################################
#     load_model = MLPClassifier(hidden_layer_sizes=[152, 14], activation='tanh', solver='adam', alpha=0.01, batch_size=1024, learning_rate='adaptive', 
#                                learning_rate_init=0.00000000001, power_t=0.5, max_iter=300, shuffle=True, random_state=1, tol=0.0001, verbose=False, 
#                                warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
#                                epsilon=1e-09, n_iter_no_change=10, max_fun=15000)     
#     #########################################################
    
#     # model = MLPClassifier(hidden_layer_sizes = [10, 10], alpha = 5.0,
#     #        random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)
    
#     ###
#     filename = 'trained_model.sav'
#     if exists(filename):
#         load_model = pickle.load(open(filename, 'rb'));
#         scaler = pickle.load(open('Scaler.pk', 'rb'));
#         # scaler_MM = pickle.load(open('Scaler_MM.pk', 'rb'));
        
        
# #         new_df_train = pd.concat( [df_train[df_train["engagement"] == False][0:30], df_train[df_train["engagement"] == True][0:30]] )
# #         new_df_train_X = new_df_train[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
# #                                "speaker_speed", "silent_period_rate"]];
# #         new_df_train_y = new_df_train[["engagement"]];
# #         new_df_train_y["engagement"] = new_df_train_y["engagement"].apply( lambda x : 1 if x == True else 0 );
        
# #         X_trainrec, X_testrec, y_trainrec, y_testrec = train_test_split(new_df_train_X, new_df_train_y, random_state = 0)
#         # scaler_2 = scaler_2.transform(X_test) ###
#         # load_model = load_model.fit(X_train_scaledrec, y_trainrec);
        
        
        
#         # print( df_train_X[0:5] );
        
#         # X_test = sc.transform(testdata.values)
#     else:    
#         # load_model = load_model.fit(X_train_scaled, y_train);
        
#         load_model = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(128, 256, 256, 256, 128, 32), random_state=1,
#                                        alpha=0.01, batch_size=1024, learning_rate='adaptive', power_t=0.5, shuffle=True,
#                                        max_iter=300,verbose=10,learning_rate_init=0.00001, momentum=0.9, nesterovs_momentum=True, 
#                                        early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-09, 
#                                        n_iter_no_change=10, max_fun=15000);
        
#         for i in range(1000):
#             ###
#             df_train = pd.read_csv('assets/train.csv')
#             random_state = randrange(42);
#             df_train = shuffle(df_train, random_state=random_state);
#             df_train_X = df_train[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
#                            "speaker_speed", "silent_period_rate"]];
#             df_train_y = df_train[["engagement"]];
#             df_train_y["engagement"] = df_train_y["engagement"].apply( lambda x : 1.0 if x == True else 0.0 );
            
#             X_train_scaled = scaler.fit_transform(df_train_X)
            
            
#             load_model = load_model.fit(X_train_scaled, df_train_y);
#             ###

#             pickle.dump(load_model, open(filename, 'wb'));
#             pickle.dump(scaler, open('Scaler.pk', 'wb'));
#             ##########
#             load_model = pickle.load(open(filename, 'rb'));
#             scaler = pickle.load(open('Scaler.pk', 'rb'));

#             print( i, load_model.score(X_train_scaled, df_train_y) );

#             if load_model.score(X_train_scaled, df_train_y) > 0.95 :
#                 print( "load_model.score(X_train_scaled, df_train_y) > 0.95" );
#                 break;
#             ###
        
        
        
        
#         pickle.dump(load_model, open(filename, 'wb'));
#         pickle.dump(scaler, open('Scaler.pk', 'wb'));
#         # pickle.dump(scaler_MM, open('Scaler_MM.pk', 'wb'));
#     ###
    
    
    
#     ###
#     # predictions = model.predict(X_test_2);
#     # roc_auc_scores = roc_auc_score(y_test_2, predictions);
#     # print(roc_auc_scores);
#     # roc_auc_scores = roc_auc_score(y_test, predictions);
#     # print(roc_auc_scores);
#     ###
    
#     # predictions = model.predict(X_test)
    
#     df_testrec = df_test[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
#                            "speaker_speed", "silent_period_rate"]];
#     # df_testrec = df_test[["title_word_count", "document_entropy", "freshness", "easiness", "fraction_stopword_presence", "normalization_rate", 
#     #                        "speaker_speed", "silent_period_rate", "avg_speakerspeed", "avg_easiness", "avg_normalization_rate"]];
    
#     # df_testrec = df_test[["easiness", "fraction_stopword_presence", "normalization_rate", 
#     #                        "speaker_speed", "silent_period_rate"]];
    
#     # scaler = scaler.transform(df_testrec)
    
#     to_predictdf = scaler.transform(df_testrec)
#     # to_predictdf = scaler_MM.transform(to_predictdf);
    
#     # predictions = load_model.predict(scaler)
#     predictions = load_model.predict(to_predictdf)
    
#     # predictions = predictions[-2309:];
#     # predictions = np.reshape(predictions, [-1, 1]);
#     # predictions = pd.DataFrame(predictions);
#     index_integer = pd.DataFrame([]);
#     index_integer["id"] = df_test["id"].astype("int");
#     # index_integer["id"] = index_integer[index_integer["id"] <= 11548];
#     index_integer["engagement"] = index_integer.apply( lambda x : predictions[x.index] );
    
#     predictions = index_integer;
#     predictions = predictions.set_index("id");
    
# #     print(predictions)
# #     print(index_integer)
    
# #     predictions = pd.concat([index_integer, predictions], ignore_index=True)

#     # print('Breast cancer dataset')
#     # print('Accuracy of NN classifier on training set: {:.2f}'
#     # .format(load_model.score(X_train_scaled, y_train)))
#     # print('Accuracy of NN classifier on test set: {:.2f}'
#     # .format(load_model.score(X_test_scaled, y_test)))
    
#     print('Breast cancer dataset')
#     print('Accuracy of NN classifier on training set: {:.2f}'
#     .format(load_model.score(X_train_scaled, df_train_y)))
#     print('Accuracy of NN classifier on test set: {:.2f}'
#     .format(load_model.score(X_test_scaled, df_train_y)))
    
    
    
#     return predictions.iloc[:,0];
```
