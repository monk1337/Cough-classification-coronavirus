import pandas as pd

class Get_data(object):
    
    def __init__(self):
        
        pass
    
    @staticmethod
    def get_pandas_signal_feats():
        feats = pd.read_pickle('./data/ser_feats_df.pkl')
        return feats
    
    @staticmethod
    def get_preprocessed_signal():
        feats = pd.read_pickle('./data/preprocess_df')
        return feats
    
    @staticmethod
    def test_train_data():
        df_d = Get_data.get_preprocessed_signal()
        
        
        #2. Get feature set, labels, and recording IDs
        X_train = df_d.drop(['label','Id','subIdx'], 1).copy()
        y_train =  df_d['label'].copy()

        ID_train = df_d['Id']
        ID_list = ID_train.drop_duplicates()
        return {
            
                'x_train': X_train, 
                'y_train': y_train, 
                'Id_train': ID_train, 
                'ID_list': ID_list
                }
