import pandas as pd


def format_date(data):
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data



