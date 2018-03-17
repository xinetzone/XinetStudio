#encoding:utf-8

from wxpy import *
import time

command = input("欢迎使用微信拜年机器人 Made By @刘惠夏青\n输入 1 -> 自动群发好友拜年信息\n输入 2 -> 开启拜年消息自动回复\n输入 3 -> 开启二合一模式\n")
if command == '1':
	bot = Bot()
	for friends in bot.friends():
		if friends.remark_name:
			friends.send("{}新年快乐啊".format(friends.remark_name))
			time.sleep(3)
elif command == '2':
	bot = Bot()
	@bot.register(Friend, msg_types=TEXT)
	def print_messages(msg):
		if '年' in msg.text:
			return "你也是，新年快乐~"
elif command == '3':
	bot = Bot()
	@bot.register(Friend, msg_types=TEXT)
	def print_messages(msg):
		if '年' in msg.text:
			return "你也是，新年快乐~"
	for friends in bot.friends():
		if friends.remark_name:
			friends.send("{}新年快乐啊".format(friends.remark_name))
			time.sleep(3)
else:
	print("输入错误")

embed()
