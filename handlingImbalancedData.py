# Python OOP yapısı kullanılmıştır.
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss



class ImbalanceDuzenle:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def RUS(self):
        rus = RandomUnderSampler(random_state=42, replacement=True)
        X_rus, y_rus = rus.fit_resample(self.X, self.y)
        print('orjinal veri boyutu:', self.y.value_counts())
        print('RUS yapıldıktan sonra veri boyutu:', y_rus.value_counts())
        return X_rus, y_rus

    def ROS(self):
        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros = ros.fit_resample(self.X, self.y)
        print('orjinal veri boyutu', self.y.value_counts())
        print('ROS yapıldıktan sonra veri boyutu:', y_ros.value_counts())
        return X_ros, y_ros

    def TL(self):
        tl = TomekLinks(sampling_strategy='majority')
        X_tl, y_tl = tl.fit_sample(self.X, self.y)
        print('orjinal veri boyutu', self.y.value_counts())
        print('TL yapıldıktan sonra veri boyutu:', y_tl.value_counts())
        return X_tl, y_tl

    def Smote(self):
        smote = SMOTE()
        X_smote, y_smote = smote.fit_sample(self.X, self.y)
        print('orjinal veri boyutu', self.y.value_counts())
        print('TL yapıldıktan sonra veri boyutu:', y_smote.value_counts())
        return X_smote, y_smote

    def NM(self):
        nm = NearMiss()
        X_nm, y_nm = nm.fit_resample(self.X, self.y)
        print('orjinal veri boyutu', self.y.value_counts())
        print('TL yapıldıktan sonra veri boyutu:', y_nm.value_counts())
        return X_nm, y_nm

