import zmq


class Subscriber:
    def __init__(self, address, opts=None) -> None:
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.SUB)
        self._sock.connect(address)
        if opts is not None:
            for k, v in opts:
                self._sock.setsockopt(k, v)
        else:
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")

    @property
    def sock(self):
        return self._sock

    def recv_json(self):
        return self._sock.recv_json()


class Publisher:
    def __init__(self, address, opts=None):
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PUB)
        self._sock.bind(address)
        if opts is not None:
            for k, v in opts:
                self._sock.setsockopt(k, v)

    def send_json(self, msg):
        self._sock.send_json(msg)
