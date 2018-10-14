class RLTurtleException(Exception):
    def __init__(self, msg):
        super(RLTurtleException, self).__init__(msg)

class OffScreenException(RLTurtleException):
    def __init__(self):
        super(OffScreenException, self).__init__(
            msg="Turtle went offscreen."
        )