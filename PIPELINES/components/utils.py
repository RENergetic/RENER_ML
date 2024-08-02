from kfp.components import InputPath

def GetListofDict(measurements_assets_dict:dict, timestamp) -> list:

    """
    
    Returns as a list the keys of a dictionary. This is used to create the for loop in the pipeline.
    If for some reasons it fails it return an empty list. 

    """

    try:
        return list(measurements_assets_dict.keys())
    except:
        return []

def CheckTrainModel(train_check:str) -> bool:

    """
    
    Transforms the train check to a boolean that can be used in the pipeline. 

    Parameters
    ----------
    train_check: A string that should have a yes/no option. If the option is anything else it defaults to "no"
		
    Returns
    -------
    True or False depending on weather the condition is met.
    """

    from icecream import ic
    
    if train_check == "yes":
        return True
    elif train_check == "no":
        return False
    else:
        ic("Forecast not send, the option is not 'yes' or 'no', please check this, the option sent was " + train_check)
        return False

def Get_List_Assets(measurement_name, dict_assets, timestamp) -> dict:

    """
    
    Returns list asssets of a certain measurement for the for loop.
    If it fails, returns an empty list.

    """

    import json
    dict_assets = json.loads(dict_assets)


    try:
        return dict_assets[measurement_name]
    except:
        return []

def CheckSendForecast(send_forecast:str) -> bool:

    """
    
    Check function for sending forecast if something other than yes or no is inputed the result defaults to a no. 

    """

    from icecream import ic
    
    if send_forecast == "yes":
        return True
    elif send_forecast == "no":
        return False
    else:
        ic("Forecast not send, the option is not 'yes' or 'no', please check this, the option sent was " + send_forecast)
        return False

def CheckSendNotification(send_notifications_check:str) -> bool:

    """
    
    Check sending notificacions into the RENeregetic System

    """

    if send_notifications_check == "no":
        return False
    elif send_notifications_check == "yes":
        return True
    else: 
        return False

def TypeModelParser(model_name:str) -> str:
    try:
        model_name_sep = model_name.split(("_"))
        return model_name_sep[-1]
    except:
        return "No model"

def Check(option1, option2, timestamp) -> bool:
    return option1 == option2


def CheckForge(list_names, list_loads, i) -> list:
    if len(list_names) <= i:
        name_model = list_names[i-1]
    else:
        name_model = "None"
    
    if len(list_loads) <= i:
        load_model = list_loads[i-1]
    else:
        load_model = "No"
    
    return [name_model, load_model]
