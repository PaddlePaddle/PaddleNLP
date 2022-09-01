python -m paddle_serving_client.convert --dirname unimo/static \
                                        --model_filename unimo_text.pdmodel \
                                        --params_filename unimo_text.pdiparams \
                                        --serving_server unimo/serving/export_checkpoint_server \
                                        --serving_client unimo/serving/export_checkpoint_client