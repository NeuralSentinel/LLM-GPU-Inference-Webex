import os
from CXTMbot import CXTMbot

from webex_bot.commands.echo import EchoCommand
from webex_bot.webex_bot import WebexBot

# Create a Bot Object
################Commenting the boilerplate code##########################
# bot = WebexBot(teams_bot_token=os.getenv("WEBEX_TEAMS_ACCESS_TOKEN"),
#                approved_rooms=['06586d8d-6aad-4201-9a69-0bf9eeb5766e'],
#                bot_name="My Teams Ops Bot",
#                include_demo_commands=True)
################Commenting the boilerplate code##########################

bot = WebexBot("----------YOUR WEBEX TOKEN----------")

#Clear default help command
bot.commands.clear()

# Add new commands for the bot to listen out for.
bot.add_command(CXTMbot())

bot.help_command = CXTMbot()

# Call `run` for the bot to wait for incoming messages.
bot.run()