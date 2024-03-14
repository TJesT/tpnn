class KeywordArgumentNotProvided(Exception):
    def __init__(self, kwarg_name: str, target_name: str, *args: object) -> None:
        super().__init__(*args)
        self.kwarg_name = kwarg_name
        self.target_name = target_name

    def __str__(self) -> str:
        return f"Argument `{self.kwarg_name}` was not provided for `{self.target_name}`"
