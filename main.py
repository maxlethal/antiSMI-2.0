from get_news import *


def run_parsing():
	"""Собирает все свежие новости всех СМИ согласно заданного расписания"""
	agencies = pd.read_sql(f"SELECT telegram FROM agencies WHERE is_parsing is TRUE",
	                       con=asmi).telegram.to_list()
	scheduler = AsyncIOScheduler(timezone='Europe/Moscow')
	scheduler.add_job(join_all, 'cron', max_instances=10, misfire_grace_time=600, hour=7,
	                  minute=00, kwargs={'agency_list': agencies})
	scheduler.add_job(join_all, 'cron', max_instances=10, misfire_grace_time=600, hour=12,
	                  minute=00, kwargs={'agency_list': agencies})
	scheduler.add_job(join_all, 'cron', max_instances=10, misfire_grace_time=600, hour=17,
	                  minute=00, kwargs={'agency_list': agencies})
	scheduler.add_job(join_all, 'cron', max_instances=10, misfire_grace_time=600, hour=21,
	                  minute=00, kwargs={'agency_list': agencies})
	scheduler.start()

	asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
	parsing_thread = threading.Thread(target=run_parsing())
	parsing_thread.start()

