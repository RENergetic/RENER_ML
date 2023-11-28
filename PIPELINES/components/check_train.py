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