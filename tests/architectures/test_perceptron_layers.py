from tests.architectures.fixtures.layer import Layer


class TestLayer:
    def test_run_through_layer(self, const_array, const_layer, const_answer):
        assert (const_layer(const_array) == const_answer).all()

    def test_after_serialization_const(self, const_array, const_layer: Layer):
        serialized_layer = Layer.from_str(const_layer.dump())
        assert (const_layer(const_array) == serialized_layer(const_array)).all()

    def test_after_serialization_random(self, array, layer):
        serialized_layer = Layer.from_str(layer.dump())
        assert (layer(array) == serialized_layer(array)).all()
