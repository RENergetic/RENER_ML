from kfp.components import InputPath

def ProduceModel(input_name:str):
    print("Model {name} saved".format(name = input_name))