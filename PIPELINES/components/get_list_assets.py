def Get_List_Assets(measurement_name, dict_assets) -> dict:
    import json
    dict_assets = json.loads(dict_assets)
    print(measurement_name)
    print(dict_assets)
    print(type(dict_assets))
    return dict_assets[measurement_name]