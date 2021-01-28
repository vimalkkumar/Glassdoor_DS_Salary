#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:01:16 2021

@author: vimal
"""
import glassdoor_data_collection as gs_data
import pandas as pd

path = '/usr/bin/chromedriver'
dataset = gs_data.get_jobs('data scientist', 15, False, path, 15)