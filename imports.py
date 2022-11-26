import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

import requests
from bs4 import BeautifulSoup
import asyncio
import threading

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
import fasttext

import warnings

from config import db_config

warnings.filterwarnings("ignore")
fasttext.FastText.eprint = lambda x: None

model_class = fasttext.load_model("models//cat_model.ftz")

tokenizer_resume = AutoTokenizer.from_pretrained("IlyaGusev/mbart_ru_sum_gazeta")
model_resume = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/mbart_ru_sum_gazeta")

tokenizer_title = AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_headline_gen_telegram")
model_title = T5ForConditionalGeneration.from_pretrained("IlyaGusev/rut5_base_headline_gen_telegram")

black_labels = ("ДАННОЕ СООБЩЕНИЕ (МАТЕРИАЛ) СОЗДАНО И (ИЛИ) РАСПРОСТРАНЕНО ИНОСТРАННЫМ СРЕДСТВОМ МАССОВОЙ ИНФОРМАЦИИ, "
                "ВЫПОЛНЯЮЩИМ ФУНКЦИИ ИНОСТРАННОГО АГЕНТА, И (ИЛИ) РОССИЙСКИМ ЮРИДИЧЕСКИМ ЛИЦОМ, ВЫПОЛНЯЮЩИМ ФУНКЦИИ "
                "ИНОСТРАННОГО АГЕНТА",
                '  Поддержите The Village подпиской https://redefine.media/about (подробная инструкция здесь)',
                '*Власти считают иноагентом  ')

db = 'aSMI'
db_user = db_config[db]['login']
db_pwd = db_config[db]['pwd']
db_name = db_config[db]['db']
db_host = db_config[db]['host']

asmi_engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_pwd}@{db_host}/{db_name}')
