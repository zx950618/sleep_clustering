from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np


def save_analysis_to_xlsx(save_path, feature_names, pearson, spearman, center_points):
    corr_df = None
    center_points_df = None
    if pearson is not None and spearman is not None:
        corr_df = {
            "feature name": feature_names,
            "1st pearson corr": np.around(pearson["corr"]["1st Dimension"], decimals=5),
            "1st pearson p": np.around(pearson["p"]["1st Dimension"], decimals=5),
            "2nd spearman corr": np.around(spearman["corr"]["2nd Dimension"], decimals=5),
            "2nd spearman p": np.around(spearman["p"]["2nd Dimension"], decimals=5)
        }
        corr_df = pd.DataFrame(corr_df)
    if center_points is not None:
        center_points_df = {
            "feature name": feature_names
        }
        for k in center_points:
            v = np.around(center_points[k], decimals=5)
            center_points_df[k] = v
        center_points_df = pd.DataFrame(center_points_df)

    with pd.ExcelWriter(save_path + "/analysis_results.xlsx") as writer:
        if corr_df is not None:
            corr_df.to_excel(writer, sheet_name="Corr")
        if center_points_df is not None:
            center_points_df.to_excel(writer, sheet_name="Center Points")


def sample_day_to_sample_people(data, index, phone, source):
    i = 1
    pre_phone = phone[0]
    pre_source = source[0]
    sample_people = []
    sample_phone = []
    sample_source = []
    current_people = [data.values[0, :]]
    while i < index.shape[0]:
        cur_phone = phone[i]
        if cur_phone == pre_phone:
            current_people.append(data.values[i, :])
        else:
            current_people = np.array(current_people)
            current_people = np.mean(current_people, axis=0)
            sample_people.append(current_people)
            sample_phone.append(pre_phone)
            sample_source.append(pre_source)
            pre_phone = cur_phone
            pre_source = source[i]
            current_people = [data.values[i, :]]
        i += 1

    current_people = np.array(current_people)
    current_people = np.mean(current_people, axis=0)
    sample_people.append(current_people)
    sample_phone.append(pre_phone)
    sample_source.append(pre_source)

    sample_people = np.array(sample_people)
    sample_phone = np.array(sample_phone)
    sample_source = np.array(sample_source)

    return sample_people, sample_source, sample_phone
