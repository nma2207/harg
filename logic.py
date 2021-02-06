import math
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Logic:
    category = {
        1: "Нормальная работа",
        2: "Частичный разряд",
        3: "Разряд низкой энергии",
        4: "Низкотемпературный перегрев"
    }

    limit = {
        "H2": 0.012,
        "C2H4": 0.050,
        "C2H2": 0.002,
        "CO": 0.075,
    }
    def __init__(self):
        super().__init__()

    def start(self):
        clf_model_filename = "clf.sav"
        regr_model_filename = "regr.sav"
        self.clf_model  = pickle.load(open(clf_model_filename, 'rb'))
        self.regr_model = pickle.load(open(regr_model_filename, 'rb'))


    def calc_graph(self, values):
        up_5_PDZ = [0.013, 0.075, 0.05, 0.002]
        after_5_PDZ = [0.012, 0.08, 0.05, 0.002]

        up_5_DZ = [0.004, 0.05, 0.015, 0.0007]
        after_5_DZ = [0.005, 0.065, 0.015, 0.0007]



        if self.view.vars.get() == 0:
            profit = []
            conc = [x / max(up_5_PDZ) for x in values]
            conc1 = [x / max(up_5_PDZ) for x in up_5_PDZ]
            conc2 = [x / max(up_5_PDZ) for x in up_5_DZ]
            for a1, a2 in zip(conc1, conc2): profit.append((a1 + a2) / 2)
            self.view.graphMat(conc, conc1, conc2, profit)

        if self.view.vars.get() == 1:
            profit = []
            conc = [x / max(after_5_PDZ) for x in values]
            conc1 = [x / max(after_5_PDZ) for x in after_5_PDZ]
            conc2 = [x / max(after_5_PDZ) for x in after_5_DZ]
            for a1, a2 in zip(conc1, conc2): profit.append((a1 + a2) / 2)
            self.view.graphMat(conc, conc1, conc2, profit)


    def make_classification(self, df):
        clf_pred = self.clf_model.predict([df.to_numpy().reshape(420*4)])
        result = clf_pred[0]
        result_label = self.category[result]
        return result_label

    def make_forecast(self, df):
        regr_pred = self.regr_model.predict([df.to_numpy().reshape(420 * 4)])
        result = regr_pred[0]
        #print(result)
        return int(result)

    def get_prediction(self, filename):
        df = pd.read_csv(filename)
        defect_label = self.make_classification(df)
        time_forecast = self.make_forecast(df)

        self.view.set_prediction(defect_label, time_forecast)

    def set_view(self, view):
        self.view = view
