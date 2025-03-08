def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def plot_data(dataframe, x_column, y_column, title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe[x_column], dataframe[y_column], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def calculate_metrics(true_values, predicted_values):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    return {'MSE': mse, 'MAE': mae}