#!/bin/bash

# ssh user@remote -NL 8888:localhost:11955 for jupyter notebook
# ssh user@remote -NL 8888:localhost:11956 for tensorflow


docker exec -it neural_compression jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
