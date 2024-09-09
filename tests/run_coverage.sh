script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../scripts/paddle_log
$script_dir/../scripts/paddle_log
MODULE=paddlenlp
IFS=','
if [ "$#" -gt 0 ]; then
MODULE=$*
fi
coverage run --source "$MODULE" -m unittest discover
coverage report -m
coverage html
