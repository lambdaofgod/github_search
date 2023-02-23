import inspect


class parse_dict_to_dataclass:
    def __init__(self, datacls, arg_name):
        self.datacls = datacls
        self.arg_name = arg_name

    def __call__(self, func):
        def decorated_func(*args):
            arg_names = inspect.getfullargspec(func).args
            assert self.arg_name in arg_names, f"no such argument {self.arg_name}"
            arg_index = arg_names.index(self.arg_name)
            arg = args[arg_index]
            try:
                arg = self.datacls(**arg)
            except TypeError as e:
                raise TypeError(f"{self.arg_name} cannot be parsed to {self.datacls}")
            args = tuple(
                [_arg if not i == arg_index else arg for (i, _arg) in enumerate(args)]
            )
            return func(*args)

        return decorated_func
