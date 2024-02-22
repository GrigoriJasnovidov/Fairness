import pandas as pd

from fmclp.groupers import _compas_targeter, _compas_race, _lsac_grouper_race, _hsls_grouper, \
    _lsac_grouper_gpa, _loan_grouper, _loan_important_cols, _compas_redundant_cols, \
    _lsac_redundant_cols, _lsac_gender_redundant_cols


def get_data(name: str, dir='../../'):
    """Get pd.DataFrame with data for FMCLP algorithm.

    Args:
        name - name of dataset. Options: loan, compas, lsac, lsac_gender, enem, hsls
        dir - auxiliary argument; needed for testing.
    Returns:
        pd.DataFrame with preprocessed data ready for implementation of FMCLP algorithm."""
    if name == 'loan':
        loan = pd.read_csv(dir + 'data/loan_cleaned.csv')
        loan = loan[loan['loan_status'] != 'Current']
        loan['target'] = loan['loan_status'].apply(_loan_grouper)
        loan = loan[_loan_important_cols]
        loan = pd.get_dummies(loan, drop_first=True)
        return loan.rename(columns={'initial_list_status_w': 'attr'})

    elif name == 'compas':
        compas = pd.read_csv(dir + 'data/compas-scores-raw.csv').drop(_compas_redundant_cols, axis=1)
        compas['attr'] = compas['Ethnic_Code_Text'].apply(_compas_race)
        compas['target'] = compas['ScoreText'].apply(_compas_targeter)
        del compas['Ethnic_Code_Text']
        del compas['ScoreText']
        return pd.get_dummies(compas, drop_first=True)

    elif name == 'lsac':
        lsac = pd.read_csv(dir + 'data/bar_pass_prediction.csv').drop(_lsac_redundant_cols, axis=1).dropna(how='any')
        lsac['attr'] = lsac['race'].apply(_lsac_grouper_race)
        lsac['target'] = lsac['gpa'].apply(_lsac_grouper_gpa)
        del lsac['race']
        del lsac['gpa']
        return pd.get_dummies(lsac, drop_first=True)

    elif name == 'lsac_gender':
        lsac_gender = pd.read_csv(dir + 'data/bar_pass_prediction.csv').drop(_lsac_gender_redundant_cols,
                                                                             axis=1).dropna(how='any')
        lsac_gender['race'] = lsac_gender['race'].apply(_lsac_grouper_race)
        lsac_gender['target'] = lsac_gender['gpa'].apply(_lsac_grouper_gpa)
        lsac_gender['attr'] = lsac_gender[['male', 'race']].apply(sum, axis=1).apply(lambda x: min(1, x))
        del lsac_gender['male']
        del lsac_gender['race']
        del lsac_gender['gpa']
        return pd.get_dummies(lsac_gender, drop_first=True)

    elif name == 'enem':
        enem = pd.read_csv(dir + 'data/enem.csv', delimiter=';')
        enem['target'] = pd.factorize(enem['Target'])[0]
        del enem['Target']
        return enem.rename(columns={'Gender': 'attr'})

    elif name == 'hsls':
        hsls = pd.read_csv(dir + 'data/hsls.csv').drop(['Unnamed: 0', 'grade9thbin', 'grade12thbin'], axis=1)
        hsls['target'] = hsls['S1M8GRADE'].apply(_hsls_grouper)
        del hsls['S1M8GRADE']
        return hsls.rename(columns={'studentgender': 'attr'})

    else:
        raise ValueError("Argument name must be 'loan', 'compas', 'lsac', 'lsac_gender', 'enem' or 'hsls'.")
