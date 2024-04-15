import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from os.path import join


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv",
                        "-c",
                        help="Path to the csv file.")
    parser.add_argument("--txt",
                        "-t",
                        help="Path to the folder containing txt files.")
    return parser.parse_args()


def age_mapping(age):
    if age == 'Baby':
        return 0
    elif age == 'Kid':
        return 1
    elif age == 'Teenager':
        return 2
    elif age == '20-30s':
        return 3
    elif age == '40-50s':
        return 4
    else:
        return 5


if __name__ == "__main__":
    txt_path = parse_args().txt
    csv_path = parse_args().csv
    df = pd.read_csv(csv_path)
    df_dropped = df.drop(['file_name', 'height', 'width', 'bbox'], axis=1)
    df_encoded = df_dropped.copy()
    encoded_columns = ['race',
                       'masked',
                       'skintone',
                       'emotion',
                       'gender'
                       ]
    for column in encoded_columns:
        df_encoded[column] = LabelEncoder().fit_transform(df_dropped[column])
    df_encoded['age'] = df_dropped['age'].apply(age_mapping)
    df_encoded = df_encoded[['face_file_name',
                             'race',
                             'gender',
                             'age',
                             'skintone',
                             'emotion',
                             'masked']]
    train, test = train_test_split(df_encoded, test_size=0.2)
    pd.DataFrame.to_csv(train, join(txt_path, 'train.txt'), sep=' ',
                        index=False, header=False)
    pd.DataFrame.to_csv(test, join(txt_path, 'test.txt'), sep=' ',
                        index=False, header=False)
    print('Done!')
