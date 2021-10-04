import json
from bokeh.embed import server_document 
script = server_document()
print(json.dumps(script))