#!/bin/bash

cd `dirname $0`

set -e

wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/qianyan/NL2SQL.zip
unzip NL2SQL.zip >/dev/null

wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/qianyan/CSpider.zip
unzip CSpider.zip >/dev/null

wget --no-check-certificate https://bj.bcebos.com/v1/dataset-bj/qianyan/DuSQL.zip
unzip DuSQL.zip >/dev/null
