import pyreft


def reft(model):
    reft_config = pyreft.ReftConfig(
        representations={
            "layer": 15, "component": "block_output",
            # alternatively, you can specify as string component access,
            # "component": "model.layers[0].output",
            "low_rank_dimension": 4,
            "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
                                                      low_rank_dimension=4)
        }
    )

    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device("cuda")
    print(reft_model.print_trainable_parameters())

    return reft_model
