#!/bin/bash
echo "Website could be visited through port 8800"
gunicorn -w 1 -b 0.0.0.0:8800 OnlineSpellingServer:app