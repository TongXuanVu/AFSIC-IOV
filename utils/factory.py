def get_model(model_name, args):
    name = model_name.lower()
    if name == "afsic-iov":
        from models.afsic_iov import AFSIC_IoV
        return AFSIC_IoV(args)
    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. Available: 'afsic-iov'."
        )
