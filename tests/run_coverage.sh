MODULE=paddlenlp
IFS=','
if [ "$#" -gt 0 ]; then
MODULE=$*
fi
coverage run --source "$MODULE" -m unittest discover
coverage report -m
coverage html
