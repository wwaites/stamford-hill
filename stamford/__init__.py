## This is a kludge because it must happen before any
## imports of stamford.network. That happens automatically
## if using the netabc tool, but does not happen when
## simply directly importing to use the code programatically.
import netabc.command as _
