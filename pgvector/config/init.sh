#!/bin/bash

psql -U vbenchmarkusr -d vbenchmarkdb -h localhost -p 5432 -f /config/init.sql