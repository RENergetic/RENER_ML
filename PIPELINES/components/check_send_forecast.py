def CheckSendForecast(send_forecast:str) -> bool:
    from discord_webhook import DiscordWebhook
    url_disc = "https://discord.com/api/webhooks/1002537248622923816/_9XY9Hi_mjzh2LTVqnmSKXlIFJ5rgBO2b8xna5pynUrzALgtC4aXSFq89uMdlW_v-ZzT"
    message = "Forecast not send, the option is not 'yes' or 'no', please check this, the option sent was{option}".format(option = send_forecast)
    webhook = DiscordWebhook(url = url_disc, content = message)
    
    if send_forecast == "yes":
        return True
    elif send_forecast == "no":
        return False
    else:
        webhook.execute()
        return False