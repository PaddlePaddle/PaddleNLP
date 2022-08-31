python -m paddle_serving_client.convert --dirname ./export_checkpoint \
                                        --model_filename unimo_text.pdmodel \
                                        --params_filename unimo_text.pdiparams \
                                        --serving_server ./export_checkpoint_server \
                                        --serving_client ./export_checkpoint_client
