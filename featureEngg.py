import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset
def feature_process(df):
    # df = pd.read_csv('../data/creditcard.csv')
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split the fe into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# if __name__ == '__main__':
#     X_train, X_test, y_train, y_test=feature_process()
#     print(X_train)
