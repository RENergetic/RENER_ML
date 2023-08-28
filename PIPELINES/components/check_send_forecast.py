def CheckSendForecast(send_forecast:str) -> bool:
    from icecream import ic
    
    if send_forecast == "yes":
        return True
    elif send_forecast == "no":
        return False
    else:
        ic("Forecast not send, the option is not 'yes' or 'no', please check this, the option sent was " + send_forecast)
        return False