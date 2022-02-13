#!/bin/bash

# ssh user@remote -NL 8888:localhost:11946 for tensorflow

docker exec -it zero_1 tensorboard --logdir ~/code/zero-shot/src/runs --port 8008 --bind_all