from pytrends.request import TrendReq

class Trends:
    pytrends = TrendReq(hl='en-US', tz=360)
    def __init__(self):
        payload = Trends.pytrends.build_payload(kw_list=["amazon", "netflix", "facebook", "tesla"], geo = "US")
        data = Trends.pytrends.interest_over_time()
        top_charts_df = Trends.pytrends.top_charts(2018, hl='en-US', tz=300, geo='GLOBAL')
        
        print(top_charts_df.head())
        print(data)
        
        print(max(data["tesla"].tolist()))
        
Trends()