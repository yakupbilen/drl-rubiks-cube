def getNetwork(name:str):
    if name == "tuned_model":
        from networks.tuned_model import Net
        return Net
    elif name == "modelres":
        from networks.modelres import Net
        return Net
    elif name == "modelpaper":
        from networks.modelpaper import Net
        return Net
    else:
        ValueError("Invalid Network Name(Must be one of 'model','modelres' or 'modelpaper')")