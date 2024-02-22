import numpy as np


def benefit(res: dict):
    """Compute percentage of discriminated against for fair and unfair classifiers.

    Args:
        res - result of implementation of fmclp function.
    Returns:
        dictionary with percentages of discriminated people."""
    classifier = res['model']['estimator']
    features = res['model']['y_test']
    data = features.copy()
    data['label'] = res['model']['x_test']
    data['prediction'] = classifier.predict(features)
    data['fair_prediction'] = res['predictions']
    data = data[['attr', 'label', 'prediction', 'fair_prediction']]

    unfair_data_0 = data[data['attr'] == 0].drop(['attr', 'fair_prediction'], axis=1)
    unfair_discriminated_0_2 = unfair_data_0[unfair_data_0['label'] == 2]
    unfair_discriminated_0_21 = unfair_discriminated_0_2[unfair_discriminated_0_2['prediction'] == 1]
    unfair_discriminated_0_20 = unfair_discriminated_0_2[unfair_discriminated_0_2['prediction'] == 0]
    unfair_discriminated_0_1 = unfair_data_0[unfair_data_0['label'] == 1]
    unfair_discriminated_0_10 = unfair_discriminated_0_1[unfair_discriminated_0_1['prediction'] == 0]
    unfair_l_0 = len(unfair_data_0)
    udo21 = unfair_discriminated_0_21.shape[0] / unfair_l_0
    udo20 = unfair_discriminated_0_20.shape[0] / unfair_l_0
    udo10 = unfair_discriminated_0_10.shape[0] / unfair_l_0

    fair_data_0 = data[data['attr'] == 0].drop(['attr', 'prediction'], axis=1)
    fair_discriminated_0_2 = fair_data_0[fair_data_0['label'] == 2]
    fair_discriminated_0_21 = fair_discriminated_0_2[fair_discriminated_0_2['fair_prediction'] == 1]
    fair_discriminated_0_20 = fair_discriminated_0_2[fair_discriminated_0_2['fair_prediction'] == 0]
    fair_discriminated_0_1 = fair_data_0[fair_data_0['label'] == 1]
    fair_discriminated_0_10 = fair_discriminated_0_1[fair_discriminated_0_1['fair_prediction'] == 0]
    fair_l_0 = len(fair_data_0)
    fdo21 = fair_discriminated_0_21.shape[0] / fair_l_0
    fdo20 = fair_discriminated_0_20.shape[0] / fair_l_0
    fdo10 = fair_discriminated_0_10.shape[0] / fair_l_0

    udo_vector = [udo10, udo20, udo21]
    fdo_vector = [fdo10, fdo20, fdo21]
    improvement_o = np.array(udo_vector) - np.array(fdo_vector)
    unfair_downgraded = np.array(udo_vector).sum()
    fair_downgraded = np.array(fdo_vector).sum()

    return {'unfair_downgraded': unfair_downgraded,
            'fair_downgraded': fair_downgraded,
            'unfair_discriminated_losses': udo_vector,
            'fair_discriminated_losses': fdo_vector,
            'improvement_0': improvement_o}
