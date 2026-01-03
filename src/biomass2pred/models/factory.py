# src/biomass2pred/models/factory.py

import timm
def create_model(
        backbone: str,
        num_outputs: int,
        pretrained: bool = True
):
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_outputs
    )
    return model