# Import libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from bs4 import BeautifulSoup
import pandas as pd
import os
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

class SentimentAnalyze:

    html_tables = {}

    input_folder = "./input"

    vader = None
    
    def __init__(self):
        nltk.downloader.download('vader_lexicon')
        SentimentAnalyze.vader = SentimentIntensityAnalyzer()
        SentimentAnalyze.vader.lexicon.update({
            'crushes': 10,
            'beats': 5,
            'misses': -5,
            'trouble': -10,
            'falls': -100,
        })
        # For every table in the datasets folder...
        for table_name in os.listdir(SentimentAnalyze.input_folder):
            #this is the path to the file. Don't touch!
            table_path = f'./input/{table_name}'
            
            # Open as a python file in read-only mode
            table_file = open(table_path, 'r')
            
            # Read the contents of the file into 'html'
            html = BeautifulSoup(open(table_path, 'r'))
            
            # Find 'news-table' in the Soup and load it into 'html_table'
            html_table = html.find(id='news-table')
            
            # Add the table to our dictionary
            SentimentAnalyze.html_tables[table_name] = html_table

    def analyze_html(self, file_name):
        # Read one single day of headlines 
        tsla = SentimentAnalyze.html_tables[file_name]
        # Get all the table rows tagged in HTML with <tr> into 'tesla_tr'
        tsla_tr = tsla.findAll('tr')

        # For each row...

        for i, table_row in enumerate(tsla_tr):
            # Read the text of the element 'a' into 'link_text'
            link_text = table_row.a.get_text()
            # Read the text of the element 'td' into 'data_text'
            data_text = table_row.td.get_text()
            # Print the count
            print(f'{i}:')
            # Print the contents of 'link_text' and 'data_text' 
            print(link_text)
            print(data_text)
            # The following exits the loop after three rows to prevent spamming the notebook, do not touch
            if i == 3:
                break
            
    def news_parser(self):
        # Hold the parsed news into a list
        parsed_news = []
        # Iterate through the news
        for file_name, news_table in SentimentAnalyze.html_tables.items():
            # Iterate through all tr tags in 'news_table'
            for x in news_table.findAll('tr'):
                
                # Read the text from the tr tag into text
                text = x.get_text()

                # Split the text in the td tag into a list 
                date_scrape = x.td.text.split()
                headline = x.a.text

                # If the length of 'date_scrape' is 1, load 'time' as the only element
                # If not, load 'date' as the 1st element and 'time' as the second
                
                if len(date_scrape) == 1:
                    time = date_scrape[0]
                    
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]
                
                # Extract the ticker from the file name, get the string up to the 1st '_'  
                ticker = file_name.split('_')[0]
                
                # Append ticker, date, time and headline as a list to the 'parsed_news' list
                parsed_news.append([ticker, date, time, headline])
        return parsed_news

    def news_sent_analyze(self, news = None):
        if (news == None):
            news = self.news_parser()
        print(news)
        scored_news = pd.DataFrame(news, columns=['ticker', 'date', 'time', 'headline'])
        scores = scored_news['headline'].apply(SentimentAnalyze.vader.polarity_scores)
        scores_df = pd.DataFrame.from_records(scores)
        scored_news = scored_news.join(scores_df)
        # Convert the date column from string to datetime
        scored_news['date'] = pd.to_datetime(scored_news.date).dt.date
        print(scored_news.head)
                    

SentimentAnalyze().analyze_html('tsla_22sep.html')
SentimentAnalyze().news_sent_analyze()
