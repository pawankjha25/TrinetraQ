import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input

from models.trinetraModel import TrinetraTransformer


def build_trinetra_model(input_shape, embed_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.1,
                         mlp_dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = TrinetraTransformer(embed_size, num_heads, ff_dim, dropout)(x, training=True)

    # Use GlobalAveragePooling1D to reduce sequence dimensions
    x = layers.GlobalAveragePooling1D()(x)  # Ensure it outputs (batch_size, embed_size)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)  # Fully defined shape
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


def main():
    data = pd.read_csv('data/creditcard.csv')
    # Normalize the 'Time' and 'Amount' columns
    scaler = StandardScaler()
    data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

    # Features
    X = data.drop(columns=['Class'])

    # Target label
    y = data['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1],)

    # Define the model
    embed_size = 64
    num_heads = 4
    ff_dim = 128
    num_transformer_blocks = 2
    mlp_units = [128]
    model = build_trinetra_model(input_shape, embed_size, num_heads, ff_dim, num_transformer_blocks, mlp_units)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
    # model.save("trinetra_model.h5", save_format="h5")


if __name__ == '__main__':
    main()
