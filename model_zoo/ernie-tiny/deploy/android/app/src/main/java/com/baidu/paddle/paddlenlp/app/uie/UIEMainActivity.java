package com.baidu.paddle.paddlenlp.app.uie;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;

import com.baidu.paddle.paddlenlp.app.R;
import com.baidu.paddle.paddlenlp.ui.Utils;
import com.baidu.paddle.paddlenlp.ernie_tiny.RuntimeOption;
import com.baidu.paddle.paddlenlp.ernie_tiny.UIEResult;
import com.baidu.paddle.paddlenlp.ernie_tiny.UIEModel;
import com.baidu.paddle.paddlenlp.ernie_tiny.SchemaLanguage;

import java.util.HashMap;

public class UIEMainActivity extends Activity implements View.OnClickListener {
    private static final String TAG = UIEMainActivity.class.getSimpleName()
            + "[FastDeploy][UIE][Java]";
    private ImageView back;
    private ImageButton btnSettings;
    private EditText etUIEInput;
    private EditText etUIESchema;
    private EditText etUIEOutput;
    private Button btnUIEAnalysis;
    private String[] inputTexts;
    private String[] schemaTexts;

    // Call 'init' and 'release' manually later
    UIEModel predictor = new UIEModel();

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.uie_activity_main);

        // Clear all setting items to avoid app crashing due to the incorrect settings
        initSettings();

        // Check and request CAMERA and WRITE_EXTERNAL_STORAGE permissions
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }

        // Init the camera preview and UI components
        initView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Reload settings and re-initialize the predictor
        checkAndUpdateSettings();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        super.onDestroy();
    }

    private void initView() {
        // Back from setting page to main page
        back = findViewById(R.id.iv_back);
        back.setOnClickListener(this);
        // Apply UIE predict
        btnUIEAnalysis = findViewById(R.id.btn_uie_analysis);
        btnUIEAnalysis.setOnClickListener(this);
        // UIE input, schema and output texts
        etUIEInput = findViewById(R.id.et_uie_input);
        etUIESchema = findViewById(R.id.et_uie_schema);
        etUIEOutput = findViewById(R.id.et_uie_output);
        // Setting page
        btnSettings = findViewById(R.id.btn_settings);
        btnSettings.setOnClickListener(this);
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_settings:
                startActivity(new Intent(UIEMainActivity.this, UIESettingsActivity.class));
                break;
            case R.id.iv_back:
                finish();
                break;
            case R.id.btn_uie_analysis:
                extractTextsInformation();
                break;
            default:
                break;
        }

    }

    public void extractTextsInformation() {
        if (updateInputTexts() && updateSchemaTexts()) {
            // Set schema before predict
            if (predictor.setSchema(schemaTexts)) {
                // Apply Information Extraction
                HashMap<String, UIEResult[]>[] results = predictor.predict(inputTexts);
                updateOutputTexts(results);
            }
        }
    }

    public void updateOutputTexts(HashMap<String, UIEResult[]>[] results) {
        if (results == null) {
            etUIEOutput.setText("抽取结果为空");
            return;
        }
        // Merge UIEResult strings -> combinedOutputText
        String combinedOutputText = UIEResult.printResult(results);
        // Update output text view (EditText)
        etUIEOutput.setText(combinedOutputText);
    }

    public boolean updateInputTexts() {
        String combinedInputText = etUIEInput.getText().toString();
        if (combinedInputText == null || combinedInputText.length() == 0) {
            // Use default text if no custom text
            combinedInputText = getString(R.string.UIE_INPUT_TEXTS_DEFAULT);
        }
        String[] texts = combinedInputText.split("[。！!：；:;]");
        if (texts.length <= 0) {
            return false;
        }
        for (int i = 0; i < texts.length; ++i) {
            texts[i] = texts[i].trim();
        }
        // Update input texts
        inputTexts = texts;
        return true;
    }

    public boolean updateSchemaTexts() {
        String combinedSchemaText = etUIESchema.getText().toString();
        if (combinedSchemaText == null || combinedSchemaText.length() == 0) {
            // Use default schema if no custom schema
            combinedSchemaText = getString(R.string.UIE_SCHEMA_DEFAULT);
        }
        String[] schemas = combinedSchemaText.split("[,，|、:；：;]");
        if (schemas.length <= 0) {
            return false;
        }
        for (int i = 0; i < schemas.length; ++i) {
            schemas[i] = schemas[i].trim();
        }

        // Update schema texts
        schemaTexts = schemas;
        return true;
    }

    @SuppressLint("ApplySharedPref")
    public void initSettings() {
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.commit();
        UIESettingsActivity.resetSettings();
    }

    public void checkAndUpdateSettings() {
        if (UIESettingsActivity.checkAndUpdateSettings(this)) {
            String realModelDir = getCacheDir() + "/" + UIESettingsActivity.modelDir;
            Utils.copyDirectoryFromAssets(this, UIESettingsActivity.modelDir, realModelDir);

            String modelFile = realModelDir + "/" + "inference.pdmodel";
            String paramsFile = realModelDir + "/" + "inference.pdiparams";
            String vocabFile = realModelDir + "/" + "vocab.txt";
            RuntimeOption option = new RuntimeOption();
            option.setCpuThreadNum(UIESettingsActivity.cpuThreadNum);
            option.setLitePowerMode(UIESettingsActivity.cpuPowerMode);
            if (Boolean.parseBoolean(UIESettingsActivity.enableLiteFp16)) {
                option.enableLiteFp16();
            }
            predictor.init(modelFile, paramsFile, vocabFile,
                    0.3f, 128, schemaTexts, 64,
                    option, SchemaLanguage.ZH);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(UIEMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            UIEMainActivity.this.finish();
                        }
                    }).show();
        }
    }

    private void requestAllPermissions() {
        ActivityCompat.requestPermissions(
                this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                0);
    }

    private boolean checkAllPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED;
    }

}
