"""
Based on:
https://github.com/hewittwill/ECGXMLReader
"""
import array
import base64
import os
from collections import OrderedDict

import numpy as np
import xmltodict


def find_substrings(substring_list, text):
    res = [substring in text for substring in substring_list]
    return int(any(res))


class ECGXMLReader:
    """Extract voltage data from a ECG XML file"""

    def __init__(self, path):
        try:
            with open(path, "rb") as xml:
                self.ECG = xmltodict.parse(xml.read().decode("utf8"))

            self.path = path
            self.id = os.path.split(self.path)[-1]

            self.num_samples = None

            self.PatientDemographics = self.ECG["RestingECG"]["PatientDemographics"]
            self.TestDemographics = self.ECG["RestingECG"]["TestDemographics"]
            self.RestingECGMeasurements = self.ECG["RestingECG"][
                "RestingECGMeasurements"
            ]
            self.Waveforms = self.ECG["RestingECG"]["Waveform"][1]
            self.Waveforms_median = self.ECG["RestingECG"]["Waveform"][0]

            self.LeadVoltages = self.makeLeadVoltages()
            self.Age = self.getAge()
            self.Gender = self.getGender()
            self.HeartRate = self.getHeartRate()
            self.Features = self.getFeatures()
            self.Diagnosis = self.readDiagnosis()
            self.DiagnosisFeatures = self.getDiagnosisFeatures()

        except Exception as e:
            print(str(e))

    def makeLeadVoltages(self, median=False):
        num_leads = 0
        leads = {}

        if median:
            target_waveforms = self.Waveforms_median
        else:
            target_waveforms = self.Waveforms

        for lead in target_waveforms["LeadData"]:
            num_leads += 1

            lead_data = lead["WaveFormData"]
            lead_b64 = base64.b64decode(lead_data)
            lead_vals = np.array(array.array("h", lead_b64))

            leads[lead["LeadID"]] = lead_vals

        self.num_samples = len(leads["I"])

        # Augment leads to 12-leads by default
        leads["III"] = np.subtract(leads["II"], leads["I"])
        leads["aVR"] = np.add(leads["I"], leads["II"]) * (-0.5)
        leads["aVL"] = np.subtract(leads["I"], 0.5 * leads["II"])
        leads["aVF"] = np.subtract(leads["II"], 0.5 * leads["I"])

        return leads

    def getLeadVoltages(self, LeadID):
        return self.LeadVoltages[LeadID]

    def getAllVoltages(self):
        return self.LeadVoltages

    def getAge(self):
        try:
            return int(self.PatientDemographics["PatientAge"])
        except KeyError:
            return np.NaN

    def getGender(self):
        try:
            if self.PatientDemographics["Gender"] == "MALE":
                return 0
            elif self.PatientDemographics["Gender"] == "FEMALE":
                return 1
            else:
                raise KeyError
        except KeyError:
            return np.NaN

    def getHeartRate(self):
        try:
            if (
                self.RestingECGMeasurements["VentricularRate"]
                == self.RestingECGMeasurements["AtrialRate"]
            ):
                return int(self.RestingECGMeasurements["VentricularRate"])
            else:
                raise KeyError
        except KeyError:
            print(
                f"{self.id}: VentricularRate and AtrialRate do not agree. use VentricularRate"
            )
            return int(self.RestingECGMeasurements["VentricularRate"])

    def getFeatures(self):
        features = []
        selected_features = [
            "PRInterval",
            "QRSDuration",
            "QTCorrected",
            "PAxis",
            "RAxis",
            "TAxis",
        ]
        for feature in selected_features:
            try:
                features.append(int(self.RestingECGMeasurements[feature]))
            except KeyError:
                features.append(np.NaN)
        return features

    def readDiagnosis(self):
        statement_dicts = self.ECG["RestingECG"]["Diagnosis"]["DiagnosisStatement"]
        if type(statement_dicts) == list:
            statement_texts = [
                statement_dict["StmtText"].lower()
                for statement_dict in statement_dicts
                if "StmtText" in statement_dict.keys()
            ]
        else:
            statement_texts = (
                [statement_dicts["StmtText"]]
                if (type(statement_dicts) == dict)
                and ("StmtText" in statement_dicts.keys())
                else []
            )
        statement = " ".join(statement_texts)
        return statement

    def getDiagnosisFeatures(self):
        target_dict = OrderedDict(
            {
                "VP": find_substrings(
                    ["ventricular premature", "premature ventricular"], self.Diagnosis
                ),
                "AP": find_substrings(
                    ["atrial premature", "premature atrial"], self.Diagnosis
                ),
                "RBBB": find_substrings(["right bundle branch block"], self.Diagnosis),
                "LBBB": find_substrings(["left bundle branch block"], self.Diagnosis),
                "LAFB": find_substrings(
                    ["left anterior fascicular block"], self.Diagnosis
                ),
                "LPFB": find_substrings(
                    ["left posterior fascicular block"], self.Diagnosis
                ),
                "BB": find_substrings(["bifascicular block"], self.Diagnosis),
                "infarct": find_substrings(["infarct"], self.Diagnosis),
                "ischemia": find_substrings(["ischemia", "ischemic"], self.Diagnosis),
                "1ST_AV": find_substrings(["1st degree a-v"], self.Diagnosis),
                "LVH": find_substrings(
                    ["left ventricular hypertrophy"], self.Diagnosis
                ),
                "RAE": find_substrings(["right atrial enlargement"], self.Diagnosis),
                "LAE": find_substrings(["left atrial enlargement"], self.Diagnosis),
                "ER": find_substrings(["early repolarization"], self.Diagnosis),
                "LV": find_substrings(["low voltage"], self.Diagnosis),
            }
        )
        return list(target_dict.values())
