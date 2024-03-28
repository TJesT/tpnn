class TestPipableResults:
    def test_pipeable_single_piping(self, a, b, c):
        assert 14 >> a == "14"
        assert "14" >> b == ["1", "4"]
        assert ["1", "4"] >> c == 14

    def test_pipeable_multiple_piping(self, a, b, c):
        assert 14 >> a >> b == ["1", "4"]
        assert "14" >> b >> c == 14
        assert ["1", "4"] >> c >> a == "14"
        assert 14 >> a >> b >> c == 14


class TestPipelineResults:
    def test_pipeline_single_piping(self, a, b, c):
        ab = a >> b
        bc = b >> c
        ca = c >> a

        assert 14 >> ab == ["1", "4"]
        assert "14" >> bc == 14
        assert ["1", "4"] >> ca == "14"

    def test_pipeline_multiple_piping(self, a, b, c):
        ab = a >> b
        bc = b >> c
        ca = c >> a
        abc = a >> b >> c
        bca = b >> c >> a
        cab = c >> a >> b

        assert 14 >> ab >> c == 14
        assert 14 >> a >> bc == 14
        assert 14 >> abc == 14

        assert "14" >> bc >> a == "14"
        assert "14" >> b >> ca == "14"
        assert "14" >> bca == "14"

        assert ["1", "4"] >> ca >> b == ["1", "4"]
        assert ["1", "4"] >> c >> ab == ["1", "4"]
        assert ["1", "4"] >> cab == ["1", "4"]
