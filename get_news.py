import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

import requests
from bs4 import BeautifulSoup

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
import fasttext

import warnings

from mycredentials import antiSMI

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


current_engine = create_engine(f'postgresql+psycopg2://{antiSMI[1]}:{antiSMI[2]}@localhost/{antiSMI[0]}')

def article2summary(article_text: str) -> str:
	"""Делает краткое саммари из новости"""
	input_ids = tokenizer_resume(
		[article_text],
		max_length=600,
		padding="max_length",
		truncation=True,
		return_tensors="pt")["input_ids"]

	output_ids = model_resume.generate(
		input_ids=input_ids,
		no_repeat_ngram_size=4)[0]

	summary = tokenizer_resume.decode(output_ids, skip_special_tokens=True)

	return summary


def summary2title(summary: str) -> str:
	"""Делает заголовок из краткой новости"""
	input_ids = tokenizer_title(
		[summary],
		max_length=600,
		add_special_tokens=True,
		padding="max_length",
		truncation=True,
		return_tensors="pt")["input_ids"]

	output_ids = model_title.generate(
		input_ids=input_ids,
		no_repeat_ngram_size=4)[0]

	title = tokenizer_title.decode(output_ids, skip_special_tokens=True)

	return title


def make_clean_text(article: str, date: int) -> dict:
	"""Обрабатывает полученную страницу новости, извлекает нужное и помещает в создаваемый словарь"""
	soup = BeautifulSoup(article, features="lxml")

	first_a = soup.find('a')
	try:
		first_link = first_a.get('href')
		if first_link.startswith('tg://resolve?domain='):
			first_link = 'NaN'
	except AttributeError:
		first_link = 'NaN'

	text = soup.get_text()
	text = text.replace("\xa0", ' ').replace("\n", ' ')
	for label in black_labels:
		text = text.replace(label, '\n')
	short_news = article2summary(text)
	title = summary2title(short_news)
	news_dict = {'date': date, 'title': title, 'short_news': short_news, 'first_link': first_link, 'raw_news': text}

	return news_dict


def make_articles_dict(channel_name: str) -> dict:
	"""Получает страницу с новостью, отдаёт её на обработку, и завершает формирование словаря использовав полученное"""
	answer = requests.get('https://tg.i-c-a.su/json/' + channel_name)
	data = answer.json()
	messages = data['messages']

	# вытаскиваем из нашей базы статей id последней статьи текущего медиа, чтобы не обрабатывать уже отработанное
	try:
		start_id = int(
			pd.read_sql(f"SELECT url FROM news WHERE date = (SELECT max(date) FROM news WHERE agency = '{agency}')",
			            current_engine).values[0][0].split('/')[-1])
	except NameError:
		start_id = 0  # если данное СМИ парсится впервые

	# выбираем только те статьи, которые старше последней
	id_articles = [(el, messages[el]['id']) for el in range(len(messages)) if
	               messages[el]['message'] and (messages[el]['id'] > start_id)]

	# получение двух предварительных словарей.

	# для словаря draft_articles часть парсинга сообщения и даты передаётся на аутсорс в функцию make_clean_text
	draft_articles = [make_clean_text(messages[el[0]]['message'], messages[el[0]]['date']) for el in id_articles if
	                  messages[el[0]]['message']]
	# это окончательный словарь, но он может содержать пустые статьи, если пост был не текстовым
	articles_dict = {id_articles[el][1]: draft_articles[el] for el in range(len(id_articles))}

	# поэтому удаляем пустые статьи
	empty_keys = [k for k, v in articles_dict.items() if not v['raw_news']]
	for k in empty_keys:
		del articles_dict[k]

	return articles_dict


def agency2db(channel_name: str) -> pd.DataFrame:
	"""
	channel_name -> channel_dict -> pd.df(channel_news) -> wright to bd
	Получает словарь текущего СМИ по его названию и записывает его в БД, предварительно обработав
	"""
	channel_dict = make_articles_dict(channel_name)
	if channel_dict:
		df = pd.DataFrame(channel_dict).T
		df['category'] = df['short_news'].apply(
			lambda x: model_class.predict(x)[0][0].split('__')[-1])  # классифицируем fasttext-ом, достаём класс
		df = df.loc[df['category'] != 'not_news']  # удаляем новости, которые классификатор не признал новостями
		df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x))  # преобразовываем timestamp-число в дату
		df['agency'] = channel_name
		df.to_sql(name='news', con=current_engine, if_exists='append', index=True)
		return df


async def join_all(agency_list: list):
	"""Передаёт список СМИ на последовательную обработку для записи свежих новостей в базу, записывает лог"""
	start_time = pd.to_datetime("today")
	print(f'Начинаю сбор текущих новостей:\n')
	for agency in agency_list:
		print(f'Собираю {agency}...')
		try:
			agency2db(agency)
		except TypeError:
			pass
		print(f'................... complited')
	finish_time = pd.to_datetime("today")
	duration = str(pd.to_timedelta(finish_time - start_time))
	print(f'\nCбор новостей завершен в {str(datetime.now().time())}')
	print(f'Уложились за {duration}\n')
	print('-------------------------------------------------------------------------------------------')
	print(f'-------------------------------------------------------------------------------------------\n')
	make_file_backup('db.db')
