import socket

from . import KPP  # noqa
from . import calc  # noqa
from . import cesm  # noqa
from . import composite  # noqa
from . import les  # noqa
from . import model  # noqa
from . import obs  # noqa
from . import sections  # noqa
from . import utils  # noqa
from .calc import *  # noqa
from .constants import *  # noqa
from .obs import *  # noqa

OPTIONS = {}

if "darya" in socket.getfqdn():
    OPTIONS["root"] = "/home/deepak/work/pump/"
else:
    OPTIONS["root"] = "/glade/work/dcherian/pump/"
