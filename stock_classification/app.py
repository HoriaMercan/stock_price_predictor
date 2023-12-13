import streamlit as st
from data import abbrevations

all_stocks = []
for full_name in abbrevations:
    all_stocks.append(abbrevations[full_name])

stock_list = st.multiselect(
    'What stocks do you want to analyze?', all_stocks)

st.write('You selected:', stock_list)

import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
from tqdm.auto import tqdm

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
'Upgrade-Insecure-Requests': '1', 'Cookie': 'v2=1495343816.182.19.234.142', 'Accept-Encoding': 'gzip, deflate, sdch',
'Referer': "http://finviz.com/quote.ashx?t="}

def get_fundamental_data(df):
  for symbol in tqdm(df.index, desc="Processing stocks"):
    try:
      #url= ("http://finviz.com/quote.ashx?t=" + symbol. lower())
      r = requests.get("http://finviz.com/quote.ashx?t=" + symbol.lower(), headers=headers)
      soup = bs(r.content, 'html.parser')
      for m in df.columns:
        df.loc[symbol, m] = soup.find(text=m).find_next(class_='snapshot-td2').text
    except Exception as e:
        print(symbol, 'not found')
        print(e)
  return df

metric = [
    'P/B',
    'P/E',
    'Forward P/E',
    'PEG',
    'Debt/Eq',
    'EPS (ttm)',
    'EPS next 5Y',
    'Dividend %',
    'ROE',
    'ROI',
    'EPS Q/Q',
    'Insider Own'
]

df = pd.DataFrame(index=stock_list, columns=metric)
df = get_fundamental_data(df)
df = df.replace('-', 0)

df['EPS next 5Y'] = df['EPS next 5Y'].str.replace('%', '')
df['ROE'] = df['ROE'].str.replace('%', '')
df['ROI'] = df['ROI'].str.replace('%', '')
df['EPS Q/Q'] = df['EPS Q/Q'].str.replace('%', '')
df['Insider Own'] = df['Insider Own'].str.replace('%', '')
df = df.apply(pd.to_numeric, errors = 'coerce')

st.title("Collected data")
st.write(df)

st.title("Stock classification")
# Stocks that are quoted at low valuations
valuedf = df[(df['P/E'].astype(float) < 30) & (df['P/B'].astype(float) < 3)]
st.write("Stocks that are quoted at low valuations:", valuedf)

# Stocks that have demonstrated earning power
EPSdf = df[df['EPS Q/Q'].astype(float) > 30]
st.write("Stocks that have demonstrated earning power:", EPSdf)

# Stocks earning good returns on equity while employing little or no debt
equitydf = df[(df['Debt/Eq'].astype(float) < 1) & (df['ROE'].astype(float) > 10)]
st.write("Stocks earning good returns on equity while employing little or no debt:", equitydf)

# Management having substantial ownership in the business
insiderdf = df[df['Insider Own'].astype(float) > .5]
st.write("Management having substantial ownership in the business:", insiderdf)