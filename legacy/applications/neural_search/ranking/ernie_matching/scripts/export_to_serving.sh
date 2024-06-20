python export_to_serving.py \
    --dirname "output" \
    --model_filename "inference.predict.pdmodel" \
    --params_filename "inference.predict.pdiparams" \
    --server_path "serving_server" \
    --client_path "serving_client" \
    --fetch_alias_names "predict"
