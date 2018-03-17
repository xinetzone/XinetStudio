#!/usr/bin/python
# -*- coding: utf-8 -*
from wxpy import *
import time
bot = Bot()  
for friends in bot.friends():
	if friends.remark_name:
		friends.send("{}我喜欢你很久了！".format(friends.remark_name))
		time.sleep(3)