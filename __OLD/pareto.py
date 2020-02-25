from comparacaoEstatistica import getTTestR

PARAMS = []
MIN_P_VALUE = 0.1
RISK_APLHA = 0.1
CRITERIOS = ''
for param in PARAMS:
    CRITERIOS += '_'
    CRITERIOS += param


def idv1PredictGreaterThanIdv2(array1, array2):
    return False


def idv1TRiskGreaterThanIdv2(array1, array2):
    return False


def getDegradationAgainstBaseline(array1):
    return False


def getDegradationPValue(array1, array2):
    return False


def getTTestR_precision(arrResultIdv1, arrResultIdv2):
    return False


def getTTestR_risk(arrResultIdv1, arrResultIdv2):
    return False


def dominate(individuo1, individuo2, firstInd_int, secondInd_int, statisticComp):
    if individuo1 == individuo2:
        return False
    arrResultIdv1 = individuo1.getResult()
    arrResultIdv2 = individuo2.getResult()

    if CRITERIOS == 'risk' or CRITERIOS == 'precision' or CRITERIOS == 'feature':
        if CRITERIOS == 'precision':
            if idv1PredictGreaterThanIdv2(arrResultIdv1, arrResultIdv2):
                alphaPrec = getTTestR_precision(arrResultIdv1, arrResultIdv2)
                if alphaPrec == -1:
                    return False
                return alphaPrec <= MIN_P_VALUE
            else:
                return False
        if CRITERIOS == 'risk':
            if idv1TRiskGreaterThanIdv2(arrResultIdv1, arrResultIdv1):
                alphaRisk = getTTestR_risk(arrResultIdv1, arrResultIdv2)
                if alphaRisk == -1:
                    return False
                return alphaRisk <= MIN_P_VALUE
            else:
                return False
        if CRITERIOS == 'georisk':
            if individuo1['georisk'] > individuo2['georisk']:
                return True
            else:
                return False

    if CRITERIOS == "precision_feature_georisk" or CRITERIOS == "precision_feature_degradation":

        predAlpha = 0
        featAlpha = 0
        riskAlpha = 0
        ID1BetterID2_Pred = False
        ID1BetterID2_Risk = False
        ID1BetterID2_Feat = False

        if CRITERIOS == "precision_feature_georisk":

            ID1BetterID2_Pred = False
            ID1BetterID2_Risk = False
            ID1BetterID2_Feat = False

            predAlpha = getTTestR(arrResultIdv1, arrResultIdv2)
            ID1BetterID2_Pred = idv1PredictGreaterThanIdv2(arrResultIdv1, arrResultIdv2)

            if individuo1['features'] < individuo2['features']:
                ID1BetterID2_Feat = True
            else:
                if individuo1['features'] == individuo2['features']:
                    featAlpha = 1

        if CRITERIOS == "precision_feature_georisk":

            if individuo1['georisk'] == individuo2['georisk']:
                riskAlpha = 1
            if individuo1['georisk'] > individuo2['georisk']:
                ID1BetterID2_Risk = True
            else:
                ID1BetterID2_Risk = False

        if CRITERIOS == "precision_feature_degradation":
            degrIdv1 = getDegradationAgainstBaseline(arrResultIdv1)
            degrIdv2 = getDegradationAgainstBaseline(arrResultIdv2)
            riskAlpha = getDegradationPValue(arrResultIdv1, arrResultIdv2)

            if degrIdv1 > degrIdv2:
                ID1BetterID2_Risk = True

        result = False

        if predAlpha <= MIN_P_VALUE and ID1BetterID2_Pred:
            if ID1BetterID2_Feat or featAlpha == 1:
                if ID1BetterID2_Risk or riskAlpha >= MIN_P_VALUE:
                    result = True

        if ID1BetterID2_Feat and featAlpha == 0:
            if predAlpha >= MIN_P_VALUE or ID1BetterID2_Pred:
                if ID1BetterID2_Risk or featAlpha >= MIN_P_VALUE:
                    result = True

        if ID1BetterID2_Risk and riskAlpha <= MIN_P_VALUE:
            if predAlpha >= MIN_P_VALUE or ID1BetterID2_Pred:
                if ID1BetterID2_Feat or featAlpha == 1:
                    result = True

        return result
