script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../scripts/paddle_log
$script_dir/../scripts/paddle_log
python -m unittest discover -v
