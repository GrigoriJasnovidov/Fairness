# This file contains auxiliary constants and functions for data preprocessing in fmclp.get_data

def _compas_targeter(x: str):
    if x == 'Low':
        return 0
    elif x == 'Medium':
        return 1
    else:
        return 2


def _compas_race(x: str):
    return int(x != 'Caucasian')


def _lsac_grouper_race(x: int):
    return int(x == 7)


def _lsac_grouper_gpa(x: float):
    if x > 3.4:
        return 2
    elif x < 3.1:
        return 0
    else:
        return 1


def _loan_grouper(x: str):
    if x == 'Fully Paid':
        return 0
    elif x in ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']:
        return 1
    else:
        return 2


def _hsls_grouper(x: int):
    if x == 0:
        return 0
    elif x == 0.2:
        return 1
    else:
        return 2


_loan_important_cols = ['loan_amnt', 'term', 'int_rate', 'verification_status', 'initial_list_status', 'target',
                        'sub_grade', 'home_ownership', 'purpose', 'dti', 'revol_bal', 'total_pymnt', 'total_rec_prncp']

_compas_redundant_cols = ['Person_ID', 'AssessmentID', 'Case_ID', 'LastName', 'MiddleName',
                          'FirstName', 'RawScore', 'DecileScore', 'IsCompleted', 'IsDeleted', 'AssessmentReason',
                          'RecSupervisionLevelText', 'DisplayText', 'Screening_Date', 'DateOfBirth']

_lsac_redundant_cols = ['ID', 'race1', 'race2', 'sex', 'bar', 'dnn_bar_pass_prediction',
                        'pass_bar', 'indxgrp2', 'gender', 'grad', 'Dropout', 'fulltime',
                        'lsat', 'zfygpa', 'ugpa', 'zgpa', 'other', 'asian', 'black', 'hisp']

_lsac_gender_redundant_cols = ['ID', 'race1', 'race2', 'sex', 'bar', 'dnn_bar_pass_prediction',
                               'pass_bar', 'indxgrp2', 'gender', 'grad', 'Dropout', 'fulltime',
                               'lsat', 'zfygpa', 'ugpa', 'zgpa', 'other', 'asian', 'black', 'hisp',
                               'bar1', 'bar1_yr', 'bar2', 'bar2_yr']
