class RLTurtleException(Exception):
    def __init__(self, msg):
        super(RLTurtleException, self).__init__(msg)

class OffScreenException(RLTurtleException):
    def __init__(self):
        super(OffScreenException, self).__init__(
            msg="Turtle went offscreen."
        )

class UserConditionException(RLTurtleException):
    def __init__(self, fn, x, y, deg):
        super(UserConditionException, self).__init__(
            msg="User condition hit at ({},{}) heading {}: {}".format(
                x, y, deg, fn.__name__
            )
        )