import glassdoor_data_collection as gs_data
import pandas as pd

path = '/usr/bin/chromedriver'
dataset = gs_data.get_jobs('data scientist', 15, False, path, 15)