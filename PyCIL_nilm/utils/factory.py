def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "bic":
        from models.bic import BiC
        return BiC(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "il2a":
        from models.il2a import IL2A
        return IL2A(args)
    elif name == "acil":
        from models.acil import ACIL
        return ACIL(args)
    else:
        assert 0
