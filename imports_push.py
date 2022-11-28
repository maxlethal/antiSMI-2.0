import numpy as np
import pandas as pd

import re
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy import create_engine

from sklearn.cluster import AgglomerativeClustering
from navec import Navec

"""
Превращение слов в эмбеддинги осуществляется c помощью navec (часть NLP-проекта natasha, 250 000 слов, эмб длиной 300), 
обученными на новостном корпусе русскоязычных текстов. Это Glove-эмбеддинги, уменьшенные с помощью квантизации.
Navec покрывает 98% слов в новостных статьях, проблема OOV решается с помощью спецэмбеддинга <unk>.
Эмбеддинг предложений - среднее эмбеддингов его слов, что хорошо работает для кластеризации.
"""
path = 'models//navec.tar'
navec = Navec.load(path)

parse_time_dict = \
	{1: {'start': '21:00:00', 'finish': '06:59:59'},
	 2: {'start': '07:00:00', 'finish': '11:59:59'},
	 3: {'start': '12:00:00', 'finish': '16:59:59'},
	 4: {'start': '17:00:00', 'finish': '20:59:59'}}

engine = sqlalchemy.create_engine('sqlite:///db.db')

pd.set_option('max_colwidth', 120)
pd.set_option('display.width', 500)
pd.set_option('mode.chained_assignment', None)

from apscheduler.schedulers.asyncio import AsyncIOScheduler


from config import db_config


db = 'aSMI'
db_user = db_config[db]['login']
db_pwd = db_config[db]['pwd']
db_name = db_config[db]['db']
db_host = db_config[db]['host']

asmi_engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_pwd}@{db_host}/{db_name}')

# pd.set_option('max_colwidth', 120)
# pd.set_option('display.width', 500)
# pd.set_option('mode.chained_assignment', None)
