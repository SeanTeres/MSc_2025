from utils import load_config
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score


def calculate_agreement_cases(df, col1, col2, relevant_terms):
    results = {}

    def to_set(findings_str):
        if isinstance(findings_str, str):
            return set(findings_str.split(';'))
        else:
            return set()

    df[f'{col1}_set'] = df[col1].apply(to_set)
    df[f'{col2}_set'] = df[col2].apply(to_set)

    print(df[f'{col1}_set'])

    # Case 1: All cases considered
    radiologist_case1 = df[f'{col1}_set'].map(lambda x: any(term in x for term in relevant_terms))  # df.apply(lambda row: any(term in row["strFindingsSimplified1_set"] for term in relevant_terms))
    panel_case1 = df[f'{col2}_set'].map(lambda x: any(term in x for term in relevant_terms)) # df.apply(lambda row: any(term in row[f'{col2}_set'] for term in relevant_terms))

    results['case1'] = calculate_metrics(radiologist_case1, panel_case1)

    # Case 2: Only when Col1 (Radiologist) is not blank (perhaps the panel only sees cases the radiologist diagnoses something in)
    df_case2 = df[df[f'{col1}_set'] != set()]
    radiologist_case2 = df_case2[f'{col1}_set'].map(lambda x: any(term in x for term in relevant_terms))
    panel_case2 = df_case2[f'{col2}_set'].map(lambda x: any(term in x for term in relevant_terms))

    results['case2'] = calculate_metrics(radiologist_case2, panel_case2)

    # Case 3: Only when Col2 (Panel) is not blank (perhaps the panel just did not see some cases)
    df_case3 = df[df[f'{col2}_set'] != set()]
    radiologist_case3 = df_case3[f'{col1}_set'].map(lambda x: any(term in x for term in relevant_terms))
    panel_case3 = df_case3[f'{col2}_set'].map(lambda x: any(term in x for term in relevant_terms))

    results['case3'] = calculate_metrics(radiologist_case3, panel_case3)

    return results

def calculate_metrics(first_opinion, second_opinion):
    accuracy = accuracy_score(first_opinion, second_opinion)
    sensitivity = recall_score(first_opinion, second_opinion)
    tn = sum((~first_opinion) & (~second_opinion))
    specificity = tn / (tn + sum(second_opinion & (~first_opinion))) if (tn + sum(second_opinion & (~first_opinion))) > 0 else 0
    count = len(first_opinion)
    return {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'samples': count}


if __name__ == "__main__":
    config = load_config()

    csv = pd.read_csv(config["silicosis_v2"]["csvpath"])

    relevant_terms = ["tbu"]

    print(calculate_agreement_cases(csv, "strFindingsSimplified1", "strFindingsSimplified2", relevant_terms))
