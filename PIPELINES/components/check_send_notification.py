def CheckSendNotification(send_notifications_check:str) -> bool:
    if send_notifications_check == "no notifications":
        return False
    else: 
        return True