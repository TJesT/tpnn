from tests.architectures.fixtures.layer import FCLayer, ACTLayer


class TestLayer:
    def test_fc_run_through_layer(self, const_array, const_actlayer, fc_const_answer):
        assert (const_array >> const_actlayer == fc_const_answer).all()

    def test_fc_after_serialization_const(self, const_array, const_actlayer: FCLayer):
        serialized_layer = FCLayer.from_str(const_actlayer.dump())
        assert (const_array >> const_actlayer == const_array >> serialized_layer).all()

    def test_fc_after_serialization_random(self, fc_array, fc_layer):
        serialized_layer = FCLayer.from_str(fc_layer.dump())
        assert (fc_array >> fc_layer == fc_array >> serialized_layer).all()

    # TODO: tests for act layer
    # TODO: tests for fc+act pipeline
